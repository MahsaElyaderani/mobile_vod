"""Resume defaults without carrying machine-specific paths between machines."""
import math
import sys

from utils.runtime import load_checkpoint

# These describe the current invocation, not the saved training experiment.
LOCAL_OPTIONS = {
    "annotations", "frames_root", "val_annotations", "val_frames_root", "video_ids",
    "val_video_ids", "checkpoint_out", "resume", "resume_best_checkpoint", "checkpoint_in",
    "big_checkpoint", "fast_checkpoint", "detector_checkpoint", "backbone_weights",
    "epochs", "device", "num_workers", "prefetch_factor", "sharing_strategy",
    "no_progress", "log_every", "cudnn_benchmark", "allow_worker_count_change",
    "learning_rate", "fast_learning_rate", "backbone_init", "init_source", "random_backbones",
}


def resume_defaults(parser, args, argv=None, payload=None):
    """Only omitted training options inherit saved values; CLI overrides stay explicit."""
    if not args.resume:
        return args
    tokens = sys.argv[1:] if argv is None else argv
    explicit = set()
    for token in tokens:
        if token == "--":
            break
        if token.startswith("--"):
            parsed = parser._parse_optional(token)
            if parsed and parsed[0] is not None:
                explicit.add(parsed[0].dest)
    payload = load_checkpoint(args.resume) if payload is None else payload
    if payload.get('budget') and 'epochs' not in explicit:
        args.epochs = payload['budget']['epochs']
    saved = dict(payload.get("run_definition", {}).get("arguments", {}))
    saved.update(payload.get("resume_arguments", {}))
    if "paired_config" in payload:
        saved.update(payload["paired_config"])
        saved["train_bn"] = not payload["paired_config"].get("freeze_bn_stats", True)
    if "training_signature" in payload:
        saved.update({key: value for key, value in payload["training_signature"].items()
                      if key in ("batch_size", "clip_stride", "seed")})
    for key, value in saved.items():
        if key == "validation_interval":
            key = "big_call_interval"
        if hasattr(args, key) and key not in LOCAL_OPTIONS and key not in explicit:
            setattr(args, key, value)
    return args


def saved_arguments(args):
    saved = {key: value for key, value in vars(args).items() if key not in LOCAL_OPTIONS}
    # Keep serialized run definitions compatible with pre-rename checkpoints.
    if "big_call_interval" in saved:
        saved["validation_interval"] = saved.pop("big_call_interval")
    return saved


def restore_optimizer(optimizer, state, learning_rate=None, fast_learning_rate=None):
    """Restore moments and group options before applying explicit LR overrides."""
    for value in (learning_rate, fast_learning_rate):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("Learning rates must be finite and positive.")
    if fast_learning_rate is not None and len(state["param_groups"]) != 2:
        raise ValueError("A separate fast learning rate requires a saved two-group optimizer.")
    optimizer.load_state_dict(state)
    if learning_rate is not None:
        optimizer.param_groups[0]["lr"] = learning_rate
    if fast_learning_rate is not None:
        optimizer.param_groups[1]["lr"] = fast_learning_rate
