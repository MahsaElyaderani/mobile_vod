from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from engine.evaluator import compute_map50, postprocess_detections
from models.paired_vod import corrupt_memory
from datasets.paired_frames import PairedFrames
from paired_config import PolicyConfig, MemoryCorruption, probability
from utils.compute import profile_conv_linear, synchronize
from routing.data import data_signature
from utils.evaluation_data import load_frame, normalized_target, raw_detection_row
from routing.paired_policy import VideoContext, choose_big, forced_big
from routing.predictions import detection, filtered
from utils.paired_checkpoints import detector_fingerprint, write_json
from utils.runtime import tqdm
from utils.runtime import LOSS_KEYS, loss_postfix


@dataclass(frozen=True)
class EvaluationOptions:
    device: str = "cpu"
    score_threshold: float = 0.001
    max_detections: int = 100
    warmup_frames: int = 3
    seed: int = 0
    progress: bool = True
    compute_losses: bool = True

    def __post_init__(self):
        probability(self.score_threshold, "Evaluation score threshold")
        if self.max_detections < 1 or self.warmup_frames < 0:
            raise ValueError("Invalid detection limit or warmup count.")


def refresh_records(dataset):
    dataset.routing_records = {int(r["id"]): r for v in dataset.video_ids for r in dataset.images_by_video[v]}
    return dataset


def select_paired_frames(dataset, frames_per_video=None, seed=0):
    if frames_per_video is None:
        return refresh_records(dataset)
    if frames_per_video < 1 or dataset.clip_length != 1:
        raise ValueError("Positive frame cap and single-frame evaluation dataset required.")
    rng = random.Random(seed)
    for video_id in dataset.video_ids:
        records = dataset.images_by_video[video_id]
        # Match native training: short videos consume no RNG draw; only consecutive windows qualify.
        if len(records) < frames_per_video:
            continue
        starts = []
        for start in range(len(records) - frames_per_video + 1):
            window = records[start:start + frames_per_video]
            ids = [int(record['frame_id']) for record in window if 'frame_id' in record]
            if len(ids) == len(window) and all(b == a + 1 for a, b in zip(ids, ids[1:])):
                starts.append(start)
        if not starts:
            raise ValueError(f"Video {video_id} has no {frames_per_video}-frame consecutive segment.")
        start = rng.choice(starts)
        dataset.images_by_video[video_id] = records[start:start + frames_per_video]
    dataset.clip_index = [(v, index) for v in dataset.video_ids
                          for index in range(len(dataset.images_by_video[v]))]
    dataset.paired_frames_per_video = frames_per_video
    return refresh_records(dataset)


def decode(output, model, threshold, maximum=100):
    for key in ("class_logits", "box_regression"):
        if not torch.isfinite(output[key]).all():
            raise FloatingPointError("Non-finite detector output.")
    return detection(**postprocess_detections(
        output["class_logits"], output["box_regression"], output["anchors"], num_classes=30,
        score_threshold=threshold, nms_iou_threshold=model.ssd_cfg.nms_iou_threshold,
        max_detections_per_class=maximum, max_total_detections=maximum))


def profiled_costs(model, device, warmup):
    frame = torch.zeros(1, 3, model.model_cfg.image_size, model.model_cfg.image_size, device=device)
    if model.model_cfg.backbone == "timm_mobilenet_v2":
        frame = PairedFrames(frame, torch.zeros(1, 3, 160, 160, device=device))
    state = model.initial_state(frame)
    with torch.no_grad():
        big = profile_conv_linear(model, lambda: model.forward_big(frame, state))
        small = profile_conv_linear(model, lambda: model.forward_small(frame, state))
        for _ in range(warmup):
            model.forward_big(frame, state)
            model.forward_small(frame, state)
    synchronize(device)
    return {"big": big, "small": small}


def seeded_corruption(state, config, seed, video_id, step):
    raw = f"{seed}:{video_id}:{step}".encode()
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device=state[0].device).manual_seed(value)
    return corrupt_memory(state, [True], config, generator)


def evaluate_paired(model, dataset, policy, options=None, corruption=None,
                    precision_ids=None, on_prediction=None):
    options = options or EvaluationOptions()
    corruption = corruption or MemoryCorruption(probability=1.)
    precision_ids = set(precision_ids or [])
    device = torch.device(options.device)
    if policy.mode == "small" and precision_ids:
        raise ValueError("Small-only baseline cannot honor big-model precision requests.")
    refresh_records(dataset)
    if not precision_ids.issubset(dataset.routing_records):
        raise ValueError("Precision requests include unevaluated image IDs.")
    model.to(device).eval()
    costs = profiled_costs(model, device, options.warmup_frames)
    calls, reasons = Counter(big=0, small=0), Counter()
    predictions, targets, frame_ids, routes = [], [], [], []
    elapsed_total = 0.
    loss_totals = dict.fromkeys(LOSS_KEYS, 0.)
    criterion = None
    if options.compute_losses:
        from losses.multibox_loss import MultiBoxLoss
        from models.anchors import BoxCoderScales
        cfg = model.ssd_cfg
        criterion = MultiBoxLoss(num_classes=30, match_iou_threshold=cfg.match_iou_threshold,
                                 box_coder_scales=BoxCoderScales(cfg.y_scale, cfg.x_scale,
                                                                cfg.height_scale, cfg.width_scale))
    with torch.no_grad(), tqdm(total=len(dataset.routing_records), desc=policy.mode, unit="frame",
                              disable=not options.progress) as bar:
        for video_id in dataset.video_ids:
            state, context = None, VideoContext()
            for step, record in enumerate(dataset.images_by_video[video_id]):
                target = normalized_target(dataset, record)
                synchronize(device)
                started = time.perf_counter()
                frame_cpu = load_frame(dataset, record, model.model_cfg.image_size, model.model_cfg.input_interpolation,
                                       paired=model.model_cfg.backbone == "timm_mobilenet_v2")
                change = context.observe_frame(frame_cpu)
                scene = (policy.mode == "fixed" and policy.scene_change_threshold is not None
                         and change > policy.scene_change_threshold)
                if scene and policy.reset_on_scene_change:
                    state = None
                    context.scene_reset()
                previous_source, age = context.previous_big, context.contribution_age(step)
                frame = frame_cpu[None].to(device)
                state = model.initial_state(frame) if state is None else state
                forced = forced_big(policy, step, context.last_big_call, scene, int(record["id"]) in precision_ids)
                small_output, small_state = None, None
                if policy.mode == "small" or (not forced and policy.mode == "fixed"):
                    small_output, small_state = model.forward_small(frame, state)
                    calls["small"] += 1
                use_big, call_reasons = choose_big(policy, step, context.last_big_call, scene,
                                                   int(record["id"]) in precision_ids)
                if use_big:
                    output, new_state = model.forward_big(frame, state)
                    calls["big"] += 1
                else:
                    output, new_state = small_output, small_state
                if output is None:
                    raise RuntimeError("No selected detector output.")
                threshold = options.score_threshold
                previous_prediction = decode(output, model, threshold, options.max_detections)
                selected = filtered(previous_prediction, options.score_threshold, options.max_detections)
                combined = policy.mode == "fixed"
                contributed = use_big and (not combined or policy.memory_feedback != "none")
                corrupted = False
                if contributed:
                    state = new_state
                    if combined and policy.memory_feedback == "corrupted":
                        state, corrupted_mask = seeded_corruption(state, corruption, options.seed, video_id, step)
                        corrupted = bool(corrupted_mask[0])
                elif (policy.mode == "small" and policy.memory_feedback != "none"
                      and model.paired_config.small_only_state_policy == "update"):
                    state = new_state
                context.record(use_big, step, contributed)
                synchronize(device)
                elapsed = time.perf_counter() - started
                elapsed_total += elapsed
                current_losses = None
                if criterion is not None:
                    losses = criterion(output, [target])
                    current_losses = {key: float(losses[key]) for key in LOSS_KEYS}
                    if not all(torch.isfinite(torch.tensor(value)) for value in current_losses.values()):
                        raise FloatingPointError("Non-finite evaluation loss.")
                    for key in LOSS_KEYS:
                        loss_totals[key] += current_losses[key]
                prediction = {**selected, "image_id": target["image_id"]}
                predictions.append(prediction)
                targets.append({k: target[k] for k in ("boxes", "labels", "image_id")})
                frame_ids.append(int(record["id"]))
                reasons.update(call_reasons)
                route = {"image_id": int(record["id"]), "video_id": video_id,
                         "frame_id": int(record["frame_id"]), "video_step": step,
                         "selected_model": "big" if use_big else "small", "big_called": use_big,
                         "small_executed": small_output is not None, "reasons": call_reasons, "frame_change": change,
                         "scene_change": bool(scene), "previous_prediction_big": previous_source,
                         "previous_contribution_age": age, "memory_committed": contributed or
                         (policy.mode == "small" and policy.memory_feedback != "none"
                          and model.paired_config.small_only_state_policy == "update"),
                         "memory_corrupted": corrupted, "inference_seconds": elapsed}
                routes.append(route)
                if on_prediction is not None:
                    on_prediction(record, target, prediction, route)
                bar.update(1)
                postfix = loss_postfix(current_losses, loss_totals, len(frame_ids)) if current_losses else {}
                postfix.update(big_calls=calls["big"], fps=f"{len(frame_ids)/elapsed_total:.2f}")
                bar.set_postfix(postfix)
    if not frame_ids:
        raise ValueError("No frames evaluated.")
    metrics = compute_map50(predictions, targets, num_classes=30, progress=options.progress)
    small_indices = [index for index, route in enumerate(routes) if route["selected_model"] == "small"]
    metrics["small_frame_image_ids"] = [frame_ids[index] for index in small_indices]
    metrics["small_frame_metrics"] = (compute_map50(
        [predictions[index] for index in small_indices], [targets[index] for index in small_indices],
        num_classes=30, progress=False) if small_indices else None)
    metrics["loss_computed"] = criterion is not None
    if criterion is not None:
        metrics.update({"mean_frame_" + key: value/len(frame_ids) for key,value in loss_totals.items()})
    counts = Counter(int(label) for target in targets for label in target["labels"])
    total_macs = sum(costs[name] * calls[name] for name in costs)
    metrics.update({"mode": policy.mode, "num_frames": len(frame_ids), "evaluated_image_ids": frame_ids,
                    "video_ids": list(dataset.video_ids), "frames_per_video": getattr(dataset, "paired_frames_per_video", None),
                    "class_names": dataset.label_to_name,
                    "ground_truth_counts_per_class": {i: counts[i] for i in range(1, 31)},
                    "score_threshold": options.score_threshold, "nms_iou_threshold": model.ssd_cfg.nms_iou_threshold,
                    "max_detections_per_class": options.max_detections, "max_total_detections": options.max_detections,
                    "big_call_frequency": calls["big"] / len(frame_ids), "executions": dict(calls),
                    "call_reasons": dict(reasons), "inference_seconds": elapsed_total,
                    "FPS": len(frame_ids) / elapsed_total, "device": str(device),
                    "conv_linear_MACs_per_branch": costs, "conv_linear_MACs_per_frame": total_macs / len(frame_ids),
                    "conv_linear_FLOPs_per_frame": 2 * total_macs / len(frame_ids),
                    "compute_scope": "Conv2d and Linear only; FLOPs=2*MACs. Excludes resize, BN, activations, pooling, recurrent elementwise operations, NMS and IO.",
                    "timing_scope": "Serial wall time including image decode/resize, transfer, scene check, model/router execution and postprocessing. Excludes warmup, metrics and visualization.",
                    "parameter_counts": {"total_detector": sum(p.numel() for p in model.parameters()),
                                         "big_backbone": sum(p.numel() for p in model.big.backbone.parameters()),
                                         "small_backbone": sum(p.numel() for p in model.fast_backbone.parameters()),
                                         "small_projection": sum(p.numel() for p in model.small_projection.parameters()),
                                         "shared_lstm": sum(p.numel() for p in model.big.lstm.parameters()),
                                         "shared_ssd": sum(p.numel() for p in model.big.feature_maps.parameters()) + sum(p.numel() for p in model.big.predictor.parameters())},
                    "detector_fingerprint": detector_fingerprint(model), "data_signature": data_signature(dataset),
                    "AP_definition": "Continuous precision-recall envelope AP at IoU 0.5; absent-GT classes omitted from mean."})
    return metrics, routes


def export_paired_run(model, dataset, policy, options, output_dir, corruption=None,
                      precision_ids=None, visual_options=None):
    from dataclasses import asdict
    from engine.visualizer import PairedVideoExporter
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json({"status": "running"}, output / "run_status.json")
    exporter = PairedVideoExporter(dataset, output, policy.mode, **visual_options) if visual_options else None
    try:
        with (output / "detections.jsonl").open("x") as detections, (output / "routing.jsonl").open("x") as routing:
            def save(record, target, prediction, route):
                row = raw_detection_row(record, target, prediction, dataset.label_to_name)
                row["route"] = route
                detections.write(json.dumps(row, allow_nan=False) + "\n")
                routing.write(json.dumps(route, allow_nan=False) + "\n")
                if exporter:
                    exporter(record, target, prediction, route)
            metrics, _ = evaluate_paired(model, dataset, policy, options, corruption, precision_ids, save)
        if exporter:
            exporter.close()
            write_json({"videos": exporter.video_index}, output / "videos_index.json")
        metrics.update(policy=asdict(policy), memory_corruption=asdict(corruption or MemoryCorruption(probability=1.)))
        write_json(metrics, output / "metrics.json")
        write_json({"status": "complete", "num_frames": metrics["num_frames"]}, output / "run_status.json")
        return metrics
    except BaseException as error:
        if exporter:
            exporter.close()
        write_json({"status": "failed", "error": str(error)}, output / "run_status.json")
        raise
