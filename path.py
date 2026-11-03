"""Edit paths here once per machine. No model/training settings belong here.

Configured relative paths are relative to mobile_vod, never the working directory.
Explicit command-line paths keep their original meaning and override these values.
None means unset: required arguments still error; optional features stay disabled.
Shell launchers need python3 on PATH to read this file, then use PYTHON below.
"""
from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = "/opt/homebrew/Caskroom/miniconda/base/envs/torch_env/bin/python3"

# Verified local ImageNet VID layout. Change DATASET_ROOT when moving machines.
DATASET_ROOT = Path("/Users/mahsa/PythonProject/datasets/ILSVRC2015")
TRAIN_ANNOTATIONS = DATASET_ROOT / "Annotations/imagenet_vid_train.json"
TRAIN_FRAMES = DATASET_ROOT / "Data/VID"
VAL_ANNOTATIONS = DATASET_ROOT / "Annotations/imagenet_vid_val.json"
VAL_FRAMES = DATASET_ROOT / "Data/VID"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"
MATCHED_TRAIN_SUBSET = PROJECT_ROOT / "splits/train_25pct.json"
TIMM_CACHE = WEIGHTS_DIR / "timm"

# Default validation data for training runs.
TRAIN_VALIDATION_ANNOTATIONS = VAL_ANNOTATIONS
TRAIN_VALIDATION_FRAMES = VAL_FRAMES

# Each task can select its own inputs/output without changing another experiment.
# Keys are existing CLI option names with '-' replaced by '_'.
TASK_PATHS = {
    "train_matched": dict(checkpoint_out=None, init_checkpoint=None, resume=None, video_ids=None, val_video_ids=None),
    "train": dict(checkpoint_in=None, resume=None, resume_best_checkpoint=None,
                  checkpoint_out=None, backbone_weights=None),
    "train_joint_pretrain": dict(checkpoint_out=None, resume=None, backbone_weights=None, video_ids=None),
    "train_paired_vod": dict(big_checkpoint=None, fast_checkpoint=None, backbone_weights=None,
                             resume=None, checkpoint_out=None, video_ids=None, val_video_ids=None),
    "evaluate": dict(checkpoint=None),
    "evaluate_paired_vod": dict(checkpoint=None,
                               output_dir=None, video_ids=None, precision_image_ids=None),
    "visualize_mobile_vod": dict(checkpoint=None, output_dir=None),
    "compare_paired_runs": dict(runs=None, output_dir=None),  # runs: list of paths
    "compare_visual_runs": dict(run_a=None, run_b=None, output_dir=None),
    "sample_vid_videos": dict(output=None),
    "prepare_joint_training": dict(checkpoint=None, checkpoint_out=None, big_checkpoint=None),
}

# Existing shell experiment selections; change these here when selecting another run.
EVALUATION_BIG = None  # Set a supported native V2 stage-2 checkpoint for run_evaluation.sh.
EVALUATION_SMALL = None
EVALUATION_JOINT = None
PREJOINT_PAIR_DIR = None
PREJOINT_BIG = None
PREJOINT_OUTPUT = None  # Must be a fresh directory; original overwrite protection applies.
RESUME_JOINT = None
RESUME_FROZEN_BIG = None


def resolve_path(value):
    """Normalize configured paths only; never resolve symlinks or touch the filesystem."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [resolve_path(item) for item in value]
    if not str(value).strip():
        raise ValueError("Empty path in path.py; use None for an unset path.")
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def task_paths(task):
    values = {}
    if task in ("train", "train_paired_vod", "train_joint_pretrain", "train_matched"):
        values.update(annotations=TRAIN_ANNOTATIONS, frames_root=TRAIN_FRAMES,
                      val_annotations=TRAIN_VALIDATION_ANNOTATIONS,
                      val_frames_root=TRAIN_VALIDATION_FRAMES)
    elif task in ("evaluate", "evaluate_paired_vod", "visualize_mobile_vod", "sample_vid_videos"):
        values.update(annotations=VAL_ANNOTATIONS, frames_root=VAL_FRAMES)
    elif task in ("compare_paired_runs", "compare_visual_runs"):
        values.update(frames_root=VAL_FRAMES)
    values.update(TASK_PATHS[task])
    return {key: resolve_path(value) for key, value in values.items() if value is not None}


def parse_paths(parser, task, argv=None):
    """Inject omitted path options, keeping argparse checks and explicit overrides.

    Injecting arguments (rather than setting defaults) preserves required-option and
    mutually-exclusive-group validation. No extra fields enter saved run definitions.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    explicit = set()
    options = {option: action for action in parser._actions for option in action.option_strings}
    for token in argv:
        if token == "--":
            break
        option = token.split("=", 1)[0]
        if option in options:
            explicit.add(options[option].dest)
        elif parser.allow_abbrev and option.startswith("--"):
            matches = [action for flag, action in options.items() if flag.startswith(option)]
            if len(matches) == 1:
                explicit.add(matches[0].dest)
    suppressed = set(explicit)
    for group in parser._mutually_exclusive_groups:
        if any(action.dest in explicit for action in group._group_actions):
            suppressed.update(action.dest for action in group._group_actions)
    configured = task_paths(task)
    injected = []
    for action in parser._actions:
        if action.dest in suppressed or action.dest not in configured:
            continue
        value = configured[action.dest]
        flag = next(option for option in action.option_strings if option.startswith("--"))
        if isinstance(value, list):
            injected.extend([flag, *value])
        else:
            injected.append(flag + "=" + value)
    return parser.parse_args(injected + argv)


def standalone_output(stage):
    return resolve_path(Path(RUNS_DIR) / f"stage{stage}" / "latest.pt")


def required_path(name):
    if name not in globals() or not name.isupper():
        raise ValueError(f"Unknown path setting: {name}")
    value = resolve_path(globals()[name])
    if value is None:
        raise ValueError(f"Set {name} in {__file__} before running this command.")
    return value


def resume_launcher(cli, destination):
    """Generate a relocatable launcher while preserving non-path training arguments."""
    import os
    import shlex
    data_names = {"--annotations": "TRAIN_ANNOTATIONS", "--frames-root": "TRAIN_FRAMES",
                  "--val-annotations": "VAL_ANNOTATIONS", "--val-frames-root": "VAL_FRAMES"}
    words, index, settings = [], 0, []
    while index < len(cli):
        flag = cli[index]
        if flag in data_names:
            name = data_names[flag]
            if name not in settings:
                settings.append(name)
            words += [flag, '"$' + name + '"']
            index += 2
        elif flag in ("--resume", "--checkpoint-out"):
            words += [flag, '"$CHECKPOINT"']
            index += 2
        else:
            words.append(shlex.quote(flag))
            index += 1
    # Locate the project from the launcher's directory, including external run folders.
    relative_root = os.path.relpath(PROJECT_ROOT, Path(destination).parent)
    assignments = "".join(name + '="$("$PYTHON" path.py --get ' + name + ')"\n'
                          for name in settings)
    return ('#!/usr/bin/env bash\nset -euo pipefail\n'
            'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
            'CHECKPOINT="$SCRIPT_DIR"/' + shlex.quote(Path(destination).name) + '\n'
            'cd -- "$SCRIPT_DIR"/' + shlex.quote(relative_root) + '\n'
            'PYTHON="$(python3 path.py --get PYTHON)"\n'
            + assignments + 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n'
            'exec "$PYTHON" -u train_paired_vod.py ' + ' '.join(words) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--get", required=True, metavar="SETTING")
    args = parser.parse_args()
    try:
        print(required_path(args.get))
    except ValueError as error:
        parser.error(str(error))
