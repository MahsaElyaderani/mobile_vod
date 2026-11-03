import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from path import parse_paths
import argparse
from dataclasses import asdict
import json
import time

import torch

from engine.paired_evaluator import EvaluationOptions, export_paired_run, select_paired_frames
from paired_config import PolicyConfig, MemoryCorruption, PairedConfig
from routing.data import add_data_arguments, selected_dataset, validate_native_categories, guard_held_out
from utils.paired_checkpoints import paired_from_payload, write_json
from utils.runtime import load_checkpoint
from utils.checkpoints import model_from_checkpoint
from models.paired_vod import PairedVOD
from routing.predictions import sha256_file
from utils.compute import synchronize


@torch.no_grad()
def benchmark_backbones(model, steps, warmup):
    if steps < 1 or warmup < 1:
        raise ValueError('Backbone benchmarking requires positive steps and warmup.')
    standalone = getattr(model, 'standalone', None)
    branches = {standalone or 'big': (model.big.backbone,
                model.paired_config.fast_image_size if standalone == 'small' else model.model_cfg.image_size)}
    if not standalone:
        branches['small'] = (model.fast_backbone, model.paired_config.fast_image_size)
    results = {}
    for name, (backbone, size) in branches.items():
        parameter = next(backbone.parameters())
        device = parameter.device
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('Backbone timing supports CPU and CUDA only.')
        frame = torch.zeros(1, 3, size, size, device=device, dtype=parameter.dtype)
        modes = [(child, child.training) for child in backbone.modules()]
        backbone.eval()
        try:
            for _ in range(warmup):
                backbone(frame)
            elapsed = 0.
            for _ in range(steps):
                synchronize(device)
                started = time.perf_counter()
                backbone(frame)
                synchronize(device)
                elapsed += time.perf_counter() - started
            results[name] = {'FPS': steps / elapsed, 'mean_ms': 1000 * elapsed / steps,
                             'input_shape': list(frame.shape), 'dtype': str(frame.dtype),
                             'device': str(device), 'steps': steps, 'warmup': warmup}
        finally:
            for child, training in modes:
                child.training = training
    return results


def load_evaluation_model(path, native_big=False):
    payload = load_checkpoint(path)
    if 'matched_format_version' in payload:
        from train_matched import model_from_matched
        if native_big or payload.get('stage') != 'matched_detection':
            raise ValueError('Matched evaluation requires a matched detection checkpoint, without --native-big.')
        return model_from_matched(payload), payload
    if not native_big:
        return paired_from_payload(payload, "detector"), payload
    if payload.get("stage") != "stage2" or payload.get("validation_pending"):
        raise ValueError("Native big evaluation requires a completed stage-2 checkpoint.")
    big = model_from_checkpoint(payload)
    # Reuse exactly the same decoding, AP, timing and export path. The unused
    # fast branch is deterministic and cannot change native big predictions.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = PairedVOD(big)
    signature = payload.get("training_signature")
    if signature is None:
        raise ValueError("Native checkpoint lacks its training signature.")
    metadata = {**payload, "training_signatures": [{
        "annotations_sha256": signature["annotation_sha256"],
        "video_ids": signature["train_video_ids"]}]}
    return model, metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the shared-memory VID pair: small, big or fixed.")
    add_data_arguments(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--native-big", action="store_true",
                        help="Evaluate a native stage-2 checkpoint through the matched big-only pipeline.")
    parser.add_argument("--mode", choices=("all", "small", "big", "fixed"), default="all")
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--scene-change-threshold", type=float)
    parser.add_argument("--keep-state-on-scene-change", action="store_true")
    parser.add_argument("--precision-image-ids", help="JSON list of image IDs forcing big calls in combined modes.")
    parser.add_argument("--memory-feedback", choices=("clean", "none", "corrupted"), default="clean")
    parser.add_argument("--memory-corruption-probability", type=float, default=1.)
    parser.add_argument("--memory-dropout", type=float, default=.1)
    parser.add_argument("--memory-noise-std", type=float, default=.1)
    parser.add_argument("--frames-per-video", type=int, help="Consecutive segment cap; shorter videos stay complete.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-threshold", type=float, default=.001)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--backbone-benchmark-steps", type=int, default=0,
                        help="Optional batch-1 backbone timing iterations after evaluation; 0 disables.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--draw-score-threshold", type=float, default=.3)
    parser.add_argument("--save-frames-every", type=int, default=30)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--playback-fps", type=float, default=30.)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-losses", action="store_true", help="Skip reported GT losses; AP50 is still evaluated.")
    args = parse_paths(parser, "evaluate_paired_vod", argv)
    if args.backbone_benchmark_steps < 0 or (args.backbone_benchmark_steps and args.warmup_frames < 1):
        parser.error('Backbone benchmark requires nonnegative steps and positive warmup frames.')
    if args.native_big and (args.mode != "big" or args.memory_feedback != "clean"):
        parser.error("--native-big requires --mode big, clean memory")
    return args


def main(argv=None):
    args = parse_args(argv)
    model, payload = load_evaluation_model(args.checkpoint, args.native_big)
    if getattr(model, 'standalone', None) and args.mode != 'big':
        raise ValueError('Matched standalone evaluation uses --mode big (the sole backbone, including standalone small).')
    if payload.get("validation_pending"):
        raise ValueError("Recover pending detector validation before evaluation.")
    dataset = selected_dataset(args, model.model_cfg.image_size, interpolation=model.model_cfg.input_interpolation, paired=model.model_cfg.uses_timm)
    select_paired_frames(dataset, args.frames_per_video, args.subset_seed)
    validate_native_categories(payload, dataset)
    guard_held_out(payload.get("training_signatures", []), dataset)
    precision_ids = set()
    if args.precision_image_ids:
        raw = json.loads(Path(args.precision_image_ids).read_text())
        if (not isinstance(raw, list) or any(type(v) is not int for v in raw) or len(set(raw)) != len(raw)):
            raise ValueError("Precision requests must be a JSON list of distinct integer image IDs.")
        precision_ids = set(raw)
        if not precision_ids.issubset(dataset.routing_records):
            raise ValueError("Precision requests include unevaluated image IDs.")
    options = EvaluationOptions(args.device, args.score_threshold, args.max_detections, args.warmup_frames,
                                 args.seed, not args.no_progress, not args.no_losses)
    corruption = MemoryCorruption(args.memory_corruption_probability, args.memory_dropout, args.memory_noise_std)
    modes = ("small", "big", "fixed") if args.mode == "all" else (args.mode,)
    policies = {mode: PolicyConfig(mode, args.interval,
                                   args.scene_change_threshold, not args.keep_state_on_scene_change,
                                   args.memory_feedback) for mode in modes}
    if args.mode == "small" and precision_ids:
        raise ValueError("Small-only baseline cannot honor precision requests.")
    visuals = (None if args.no_visuals else
               {"draw_score_threshold": args.draw_score_threshold, "max_side": args.max_side,
                "playback_fps": args.playback_fps, "save_frames_every": args.save_frames_every})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json({"status": "running"}, output / "run_status.json")
    provenance = {"checkpoint": str(Path(args.checkpoint).resolve()),
                  "checkpoint_sha256": sha256_file(args.checkpoint), "epoch": payload.get("epoch"),
                  "global_step": payload.get("global_step"), "budget": payload.get("budget"),
                  "native_big": args.native_big, "arguments": vars(args)}
    write_json(provenance, output / "evaluation_source.json")
    results = {}
    try:
        for mode, policy in policies.items():
            name = f"fixed_k{args.interval}" if mode == "fixed" else mode
            results[name] = export_paired_run(model, dataset,
                                              policy, options, output / name, corruption,
                                              precision_ids if mode == "fixed" else None, visuals)
            results[name]["evaluation_source"] = provenance
            if args.native_big:
                results[name]["native_detector_parameters"] = sum(p.numel() for p in model.big.parameters())
                results[name]["evaluation_wrapper_note"] = (
                    "Unused fast branch exists only for evaluator compatibility; total_detector includes it. "
                    "Use native_detector_parameters for native model size. Timed detections execute big only.")
            write_json(results[name], output / name / "metrics.json")
        write_json({"results": {name: {key: metrics[key] for key in
                    ("mAP@0.5", "FPS", "conv_linear_FLOPs_per_frame", "big_call_frequency", "num_frames",
                     "present_classes", "absent_classes", "positive_counts_per_class")}
                    for name, metrics in results.items()}, "arguments": vars(args)}, output / "summary.json")
        if args.backbone_benchmark_steps:
            timings = benchmark_backbones(model, args.backbone_benchmark_steps, args.warmup_frames)
            report = {'backbones': timings, 'torch_version': str(torch.__version__),
                      'scope': 'Serial batch-1 backbone wrapper including normalization where implemented; '
                               'resident synthetic input; excludes resize, transfer, projection, LSTM, SSD and NMS.'}
            write_json(report, output / 'backbone_benchmark.json')
            print(json.dumps(report, indent=2))
        write_json({"status": "complete", "modes": list(results)}, output / "run_status.json")
    except BaseException as error:
        write_json({"status": "failed", "error": str(error)}, output / "run_status.json")
        raise
    print(json.dumps(json.loads((output / "summary.json").read_text())["results"], indent=2))
    return results


if __name__ == "__main__":
    main()
