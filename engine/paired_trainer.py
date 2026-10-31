from dataclasses import asdict
import argparse
from utils.resume import saved_arguments, restore_optimizer
from pathlib import Path
from types import SimpleNamespace
import math
import random
import uuid

import torch

from datasets.dataloader import build_dataloader
from datasets.transforms import TrainTransform
from engine.paired_evaluator import EvaluationOptions, evaluate_paired, select_paired_frames
from losses.multibox_loss import MultiBoxLoss
from models.paired_vod import corrupt_memory
from paired_config import MemoryCorruption, PairedConfig, PolicyConfig
from routing.data import data_signature, guard_held_out, selected_dataset, validate_native_categories
from routing.paired_policy import random_training_schedule, validate_random_training_resume
from utils.checkpoints import capture_rng, restore_rng
from utils.paired_checkpoints import (paired_metadata, initialize_pair, load_pair,
                                      save_training_payload, finish_validation, run_lock)
from utils.runtime import LOSS_KEYS, load_checkpoint, loss_postfix, tqdm

def criterion_for(model):
    from models.anchors import BoxCoderScales
    s = model.ssd_cfg
    return MultiBoxLoss(num_classes=30, match_iou_threshold=s.match_iou_threshold,
                        box_coder_scales=BoxCoderScales(s.y_scale, s.x_scale, s.height_scale, s.width_scale))


def detector_batch(model, frames, targets, criterion, corruption, freeze_big_in_joint=False):
    big_mask, small_commit = random_training_schedule(
        frames.shape[0], frames.shape[1], frames.device)
    state, logits, boxes, corrupted = None, [], [], 0
    for step in range(frames.shape[1]):
        output, state = model.forward_selected(frames[:, step], state, big_mask[:, step], small_commit[:, step])
        state, affected = corrupt_memory(state, big_mask[:, step], corruption)
        corrupted += int(affected.sum())
        logits.append(output["class_logits"])
        boxes.append(output["box_regression"])
    logits, boxes = torch.stack(logits, 1), torch.stack(boxes, 1)
    if freeze_big_in_joint:
        # The frozen big detections initialize memory; supervise only the learning small path.
        selected = ~big_mask
        if not selected.any():
            # A genuine random draw may select only big frames. Do not resample
            # or take a zero-gradient optimizer step (RMSprop momentum can move weights).
            return None, {"big_selections": int(big_mask.sum()),
                          "corrupted_updates": corrupted}
        logits, boxes = logits[selected], boxes[selected]
        flat_targets = [target for clip in targets for target in clip]
        targets = [flat_targets[index] for index in selected.flatten().nonzero().flatten().tolist()]
    loss = criterion({"class_logits": logits, "box_regression": boxes,
                      "anchors": output["anchors"]}, targets)
    return loss, {"big_selections": int(big_mask.sum()), "corrupted_updates": corrupted}


def train_detector_epoch(model, loader, optimizer, criterion, args, epoch=1):
    phase = "joint"
    freeze_big = getattr(args, "freeze_big_in_joint", False)
    model.configure_detector_training(freeze_big=freeze_big)
    totals, batches, selections, corrupted = dict.fromkeys(LOSS_KEYS, 0.), 0, 0, 0
    skipped_batches = 0
    corruption = MemoryCorruption(args.memory_corruption_probability, args.memory_dropout, args.memory_noise_std)
    description = f"Epoch {epoch} {phase}{' [big/shared frozen]' if freeze_big else ''} (current/mean)"
    with tqdm(loader, desc=description, disable=args.no_progress) as bar:
        for batch_index, (frames, targets) in enumerate(bar, 1):
            frames = frames.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, info = detector_batch(model, frames, targets, criterion, corruption,
                                        freeze_big_in_joint=freeze_big)
            selections += info["big_selections"]
            corrupted += info["corrupted_updates"]
            if loss is None:
                skipped_batches += 1
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                continue
            if not torch.isfinite(loss["total_loss"]):
                raise FloatingPointError("Non-finite paired detector loss.")
            loss["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            current = {key: float(loss[key].detach()) for key in LOSS_KEYS}
            for key in LOSS_KEYS:
                totals[key] += current[key]
            batches += 1
            bar.set_postfix(loss_postfix(current, totals, batches))
            log_every = getattr(args, "log_every", 50)
            if batches == 1 or batches % log_every == 0:
                print(f"{description}, batch {batches}: " + ", ".join(
                    f"{key}={current[key]:.4f}/{totals[key]/batches:.4f}" for key in LOSS_KEYS), flush=True)
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
    if not batches:
        raise ValueError("No complete training batches.")
    print(f"Epoch {epoch} {phase} mean: " + ", ".join(
        f"{key}={totals[key]/batches:.4f}" for key in LOSS_KEYS), flush=True)
    return {**{key: value / batches for key, value in totals.items()}, "phase": phase, "optimizer_steps": batches,
            "big_selections": selections, "corrupted_updates": corrupted, "skipped_batches": skipped_batches,
            "big_shared_frozen": freeze_big}


def add_training_arguments(parser):
    from routing.data import add_data_arguments
    add_data_arguments(parser)
    parser.add_argument("--val-annotations")
    parser.add_argument("--val-frames-root")
    parser.add_argument("--val-video-ids")
    parser.add_argument("--max-val-videos", type=int)
    parser.add_argument("--val-frames-per-video", type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-length", type=int, default=17)
    parser.add_argument("--clip-stride", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--sharing-strategy", choices=("file_descriptor", "file_system"),
                        help="CPU tensor transfer strategy; file_system avoids per-storage cached file descriptors.")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--grad-clip", type=float, default=10.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--big-call-interval", type=int, default=8,
                        help="Call the big backbone every N frames during fixed-policy validation; not validation frequency.")
    parser.add_argument("--validation-interval", dest="big_call_interval", type=int,
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Print a persistent loss line every N detector batches, plus first batch and epoch mean.")


def build_training_data(args, model, metadata):
    if (min(args.epochs, args.batch_size, args.clip_length, args.clip_stride, args.prefetch_factor,
            args.big_call_interval) < 1 or args.num_workers < 0 or
            args.max_batches is not None and args.max_batches < 1 or
            not math.isfinite(args.grad_clip) or args.grad_clip <= 0):
        raise ValueError("Invalid training controls.")
    if bool(args.val_annotations) != bool(args.val_frames_root):
        raise ValueError("Supply both validation annotations and frames root.")
    print(f"Loading training annotations and indexing clips: {args.annotations}", flush=True)
    train = selected_dataset(args, model.model_cfg.image_size, args.clip_length, args.clip_stride, interpolation=model.model_cfg.input_interpolation, paired=model.model_cfg.backbone == "timm_mobilenet_v2")
    print(f"Training index ready: {len(train.video_ids)} videos, {len(train)} clips.", flush=True)
    train.transform = TrainTransform(model.model_cfg.image_size, interpolation=model.model_cfg.input_interpolation,
                                     paired=model.model_cfg.backbone == "timm_mobilenet_v2")
    if not metadata:
        metadata.update(category_id_to_label=train.category_id_to_label, label_to_name=train.label_to_name,
                        ssd_config=asdict(model.ssd_cfg))
    validate_native_categories(metadata, train)
    if len(train) < args.batch_size:
        raise ValueError("Training selection has fewer complete clips than a full batch.")
    val = None
    if args.val_annotations:
        print(f"Loading validation annotations: {args.val_annotations}", flush=True)
        val_args = SimpleNamespace(annotations=args.val_annotations, frames_root=args.val_frames_root,
                                   video_ids=args.val_video_ids, max_videos=args.max_val_videos, subset_seed=args.subset_seed)
        val = selected_dataset(val_args, model.model_cfg.image_size, interpolation=model.model_cfg.input_interpolation, paired=model.model_cfg.backbone == "timm_mobilenet_v2")
        select_paired_frames(val, args.val_frames_per_video, args.subset_seed)
        validate_native_categories(metadata, val)
        guard_held_out([data_signature(train)], val)
    return train, val


def make_loader(dataset, args, shuffle=True, drop_last=True):
    return build_dataloader(dataset, args.batch_size, shuffle, args.num_workers,
                            pin_memory=torch.device(args.device).type == "cuda", drop_last=drop_last,
                            prefetch_factor=args.prefetch_factor,
                            sharing_strategy=getattr(args, "sharing_strategy", None))


def validation(model, dataset, args):
    rng = capture_rng()
    try:
        policy = PolicyConfig(mode="fixed", interval=args.big_call_interval)
        metrics, _ = evaluate_paired(model, dataset, policy,
                                     EvaluationOptions(args.device, warmup_frames=0, progress=not args.no_progress))
        if getattr(args, "diagnostic_validation", False):
            diagnostics = {}
            for name, diagnostic_policy in (
                    ("big_only", PolicyConfig(mode="big")),
                    ("small_only", PolicyConfig(mode="small")),
                    ("small_zero_memory", PolicyConfig(mode="small", memory_feedback="none")),
                    ("fixed_no_memory", PolicyConfig(mode="fixed", interval=args.big_call_interval,
                                                     memory_feedback="none"))):
                result, _ = evaluate_paired(model, dataset, diagnostic_policy,
                    EvaluationOptions(args.device, warmup_frames=0, progress=not args.no_progress))
                diagnostics[name] = result
            metrics["diagnostics"] = diagnostics
        return metrics
    finally:
        restore_rng(rng)


def recover_validation(payload, model, val_dataset, args):
    if not payload.get("validation_pending"):
        return payload
    if val_dataset is None:
        raise ValueError("Pending validation requires the saved validation selection.")
    from utils.runtime import atomic_save_checkpoint, epoch_checkpoint_path
    numbered = epoch_checkpoint_path(args.checkpoint_out, payload["epoch"])
    if numbered.exists():
        completed = load_checkpoint(numbered)
        if (completed.get("run_id") != payload["run_id"] or completed.get("epoch") != payload["epoch"] or
                completed.get("run_definition") != payload["run_definition"]):
            raise FileExistsError("Numbered checkpoint belongs to another run.")
        if not completed.get("validation_pending"):
            if completed["model"].keys() != payload["model"].keys() or any(
                    not torch.equal(completed["model"][key], payload["model"][key]) for key in payload["model"]):
                raise ValueError("Completed validation checkpoint contains different detector weights.")
            atomic_save_checkpoint(completed, args.checkpoint_out)
            return completed
    metrics = validation(model, val_dataset, args)
    return finish_validation(payload, metrics, args.checkpoint_out)


def train_pair(args):
    print("Initializing paired training and loading checkpoint...", flush=True)
    if args.log_every < 1:
        raise ValueError("--log-every must be positive.")
    if not math.isfinite(args.rmsprop_eps) or args.rmsprop_eps <= 0:
        raise ValueError("RMSprop epsilon must be finite and positive.")
    if args.fast_learning_rate is not None and (not math.isfinite(args.fast_learning_rate) or args.fast_learning_rate <= 0):
        raise ValueError("Fast learning rate must be finite and positive.")
    if args.reset_fast_bn and args.fast_bn_momentum is None and not args.train_bn:
        raise ValueError("Reset fast BN requires --fast-bn-momentum or --train-bn to learn running stats.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    stage = "detector"
    resume = load_checkpoint(args.resume) if args.resume else None
    if resume:
        validate_random_training_resume(resume)
    destination = Path(args.checkpoint_out).resolve()
    best = destination.with_name(destination.stem + "_best" + (destination.suffix or ".pt"))
    if not resume and (destination.exists() or best.exists() or list(destination.parent.glob(destination.stem + "_epoch_*"))):
        raise FileExistsError("Use a new checkpoint destination or --resume.")
    if resume:
        model, payload = load_pair(args.resume, stage)
        metadata, sources = payload, payload["sources"]
        if args.backbone is not None and args.backbone != model.model_cfg.backbone:
            raise ValueError('Backbone family changed on resume.')
        if args.big_width_mult is not None and args.big_width_mult != model.model_cfg.width_mult:
            raise ValueError('Big width changed on resume.')
        expected = model.paired_config
        if any(getattr(args, key) != getattr(expected, key) for key in
               ("fast_width_mult", "fast_image_size", "small_only_state_policy", "fast_bn_momentum")) or args.train_bn == expected.freeze_bn_stats:
            raise ValueError("Paired model configuration changed on resume.")
        for existing in (destination, best):
            if existing.exists():
                stored = load_checkpoint(existing)
                if stored.get("run_id") != resume["run_id"] or stored.get("run_definition") != resume["run_definition"]:
                    raise FileExistsError("Resume destination belongs to a different run.")
                if stored["epoch"] > resume["epoch"]:
                    raise ValueError("A newer checkpoint already exists; resume the latest checkpoint.")
        from routing.predictions import sha256_file
        if args.big_checkpoint and sha256_file(args.big_checkpoint) != sources["big_checkpoint_sha256"]:
            raise ValueError("--big-checkpoint differs from the resumed initialization source.")
        if args.fast_checkpoint and sha256_file(args.fast_checkpoint) != sources.get("fast_checkpoint_sha256"):
            raise ValueError("--fast-checkpoint differs from the resumed initialization source.")
    else:
        if not args.big_checkpoint and args.init_source != "fresh":
            raise ValueError("New joint_detection training requires --init-checkpoint (legacy alias: --big-checkpoint).")
        cfg = PairedConfig(fast_width_mult=args.fast_width_mult, fast_image_size=args.fast_image_size,
                           small_only_state_policy=args.small_only_state_policy, freeze_bn_stats=not args.train_bn,
                           fast_bn_momentum=args.fast_bn_momentum)
        from utils.joint_initialization import make_joint_config
        fresh_config = (make_joint_config(args.backbone or 'mobilenet_v1',
                                          1. if args.big_width_mult is None else args.big_width_mult)
                        if args.init_source == 'fresh' else None)
        model, metadata, sources = initialize_pair(args.big_checkpoint, cfg, args.fast_init, args.fast_checkpoint,
                                                   init_source=args.init_source, model_config=fresh_config,
                                                   backbone_weights=args.backbone_weights, random_backbones=args.random_backbones)
        if args.reset_fast_bn:
            model.reset_fast_bn_stats()
    print(f"Moving detector to {args.device}...", flush=True)
    model.to(args.device)
    train, val = build_training_data(args, model, metadata)
    print("Building training data signature and checking held-out validation...", flush=True)
    own_signature = data_signature(train)
    signatures = list(metadata.get('training_signatures', []))
    if own_signature not in signatures:
        signatures.append(own_signature)
    if val is not None:
        guard_held_out(signatures, val)
    definition = {"stage": stage, "training_data": own_signature,
                  "validation_data": data_signature(val) if val else None,
                  "arguments": {key: value for key, value in vars(args).items()
                                if key not in ("epochs", "resume", "checkpoint_out", "no_progress", "big_checkpoint",
                                               "fast_checkpoint", "backbone_weights", "sharing_strategy", "log_every", "init_source")},
                  "model": paired_metadata(model), "sources": sources}
    definition["random_schedule_version"] = 3
    # Preserve the checkpoint schema used by existing runs.
    definition["arguments"]["validation_interval"] = definition["arguments"].pop("big_call_interval")
    # New opt-in controls should not invalidate legacy run definitions at their defaults.
    defaults = {"backbone": None, "big_width_mult": None, "random_backbones": False, "fast_bn_momentum": None, "reset_fast_bn": False, "fast_learning_rate": None,
                "rmsprop_eps": 1., "diagnostic_validation": False, "freeze_big_in_joint": False}
    for key, default in defaults.items():
        if definition["arguments"].get(key) == default:
            definition["arguments"].pop(key, None)
    #if resume and not (
    #        definition.get("training_data") == resume["run_definition"].get("training_data")
    #        and definition.get("validation_data") == resume["run_definition"].get("validation_data")
    #        and definition.get("stage") == resume["run_definition"].get("stage")
    #):
    #    raise ValueError("Resume configuration or training/validation data differ from the saved run.")

    criterion = criterion_for(model)
    loader = make_loader(train, args)
    if not model.paired_config.freeze_bn_stats and args.batch_size < 2:
        raise ValueError("Training BatchNorm statistics requires batch size >=2.")
    lr = args.learning_rate if args.learning_rate is not None else 1e-4
    parameters = model.parameters()
    if (len(resume["optimizer"]["param_groups"]) == 2 if resume else args.fast_learning_rate is not None):
        parameters = [{"params": model.big.parameters(), "lr": lr},
                      {"params": list(model.fast_backbone.parameters()) + list(model.small_projection.parameters()),
                       "lr": args.fast_learning_rate if args.fast_learning_rate is not None else lr}]
    # Include frozen parameters so optimizer groups stay stable across phases and resume.
    optimizer = torch.optim.RMSprop(parameters, lr=lr, alpha=.9, momentum=.9, eps=args.rmsprop_eps)
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    if resume:
        restore_optimizer(optimizer, resume["optimizer"], args.learning_rate,
                          args.fast_learning_rate)
        # Keep run identity stable; effective invocation settings are saved separately.
        definition = resume["run_definition"]
        restore_rng(resume["rng_state"])
        payload = recover_validation(resume, model, val, args)
        start, best_map = payload["epoch"], payload.get("best_map50")
        history, run_id = payload["history"], payload["run_id"]
    else:
        start, best_map, history, run_id = 0, None, [], uuid.uuid4().hex
    if start > args.epochs:
        raise ValueError("--epochs is a total epoch target and is below the saved epoch.")
    if start == args.epochs:
        tqdm.write(f"Epoch target {args.epochs} already reached; no additional training required.")
        return payload
    if resume:
        # Carry the actual historical best when relocating an intact run directory.
        source = Path(args.resume)
        source_best = source.with_name(source.stem + "_best" + (source.suffix or ".pt"))
        if not best.exists() and source_best.exists():
            historical = load_checkpoint(source_best)
            if (historical.get("run_id") != run_id or
                    historical.get("run_definition") != resume["run_definition"] or
                    historical.get("validation_pending") or historical["epoch"] > start or
                    historical.get("validation_metrics", {}).get("mAP@0.5") != best_map):
                raise ValueError("Historical best does not match the resumed run and best metric.")
            from utils.runtime import atomic_save_checkpoint
            atomic_save_checkpoint(historical, best, overwrite=False)
    usable = len({video_id for video_id,_ in train.clip_index})
    tqdm.write(f"{stage}: {len(train.video_ids)} selected videos ({usable} with complete clips), {len(train)} clips; {len(loader)} full batches/epoch; "
               f"{len(train)%args.batch_size} clips dropped. BN stats frozen: {model.paired_config.freeze_bn_stats}.")
    for epoch in range(start + 1, args.epochs + 1):
        # Recreate workers each epoch so a resumed run uses the same RNG-derived worker seeds.
        loader = make_loader(train, args)
        stats = train_detector_epoch(model, loader, optimizer, criterion, args, epoch)
        history.append({"epoch": epoch, **stats})
        payload = {**paired_metadata(model), "stage": stage, "model": model.state_dict(),
                   "epoch": epoch, "optimizer": optimizer.state_dict(), "rng_state": capture_rng(),
                   "category_id_to_label": train.category_id_to_label, "label_to_name": train.label_to_name,
                   "sources": sources, "training_signatures": signatures, "run_definition": definition,
                   "run_id": run_id, "checkpoint_out": str(destination), "history": history,
                   "best_map50": best_map, "validation_pending": val is not None,
                   "resume_arguments": saved_arguments(args)}
        payload["resume_arguments"]["learning_rate"] = optimizer.param_groups[0]["lr"]
        if len(optimizer.param_groups) == 2:
            payload["resume_arguments"]["fast_learning_rate"] = optimizer.param_groups[1]["lr"]
        save_training_payload(payload, destination)
        if val is not None:
            try:
                payload = finish_validation(payload, validation(model, val, args), destination)
            except BaseException:
                # The already-published pending checkpoint retains model, optimizer and RNG state.
                raise
            best_map = payload["best_map50"]
        tqdm.write(f"Saved {stage} epoch {epoch}: {destination}")
    return payload if resume or args.epochs else None


def locked_training(args):
    previous_strategy = torch.multiprocessing.get_sharing_strategy()
    try:
        with run_lock(args.checkpoint_out):
            return train_pair(args)
    finally:
        torch.multiprocessing.set_sharing_strategy(previous_strategy)
