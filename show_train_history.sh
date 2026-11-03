#!/usr/bin/env bash
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vision
set -u
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

RUNS="$(python path.py --get RUNS_DIR)"
ROOT="${1:-$RUNS}"

python - "$ROOT" <<'PY' | tee "$RUNS/training_history.txt"
import sys
import math
from pathlib import Path
from utils.runtime import load_checkpoint

def show_history(checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path)
    stage = checkpoint.get("stage")
    if stage not in ("matched_classification", "matched_detection"):
        return
    task = stage.removeprefix("matched_")
    metric = checkpoint.get("selection_metric", "micro_ap" if task == "classification" else "mAP@0.5")
    budget = checkpoint.get("budget", {})
    print(f"\n{'=' * 65}\n{checkpoint.get('experiment')}: {task}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved epoch: {checkpoint.get('epoch')}")
    print(f"Updates: {budget.get('updates')} / {budget.get('total_updates')}")
    if 'total_updates' in budget and 'updates' in budget:
        print(f"Remaining updates: {budget['total_updates'] - budget['updates']}")
    for index, group in enumerate(checkpoint.get('optimizer', {}).get('param_groups', [])):
        print(f"Saved learning rate (group {index}): {group['lr']:.10g}")

    origin = budget.get("origin_updates")
    if origin is not None:
        print(f"Continuation starts after update: {origin}")
    history = checkpoint.get("history", [])
    for row in history:
        if origin is not None:
            print("\n[New subset continuation]" if row.get("update", 0) > origin else "\n[Inherited history]")
        print(f"\nEpoch {row.get('epoch')}, update {row.get('update', 'not recorded')}")
        print("Training:", row.get("train", "not recorded"))
        validation = row.get("validation", {})
        for key in ("micro_ap", "bce_loss") if task == "classification" else ("mAP@0.5",):
            score = validation.get(key)
            if score is not None:
                print(f"Validation {key}: {score:.6f}" + (f" ({100 * score:.2f}%)" if key != "bce_loss" else ""))
        present = validation.get("present_classes")
        if present is not None:
            print(f"Class coverage: {len(present)}/30")

    scored = [r for r in history if metric in r.get("validation", {})]
    if scored:
        best = (min if metric == "bce_loss" else max)(scored, key=lambda r: r["validation"][metric])
        print(f"\nBest validation {metric}: {best['validation'][metric]:.6f} "
              f"at epoch {best.get('epoch')}, update {best.get('update')}")
    metric = "micro_ap" if task == "classification" else "mAP@0.5"
    scored = [r for r in history if r.get("validation", {}).get(metric) is not None
              and math.isfinite(r['validation'][metric])]
    best = max(scored, key=lambda r: r['validation'][metric]) if scored else None
    return (str(checkpoint_path.parent.relative_to(root)), checkpoint.get('experiment', '?'), metric,
            f"{100 * best['validation'][metric]:.2f}%" if best else 'not recorded',
            str(best.get('epoch', '-')) if best else '-', str(best.get('update', '-')) if best else '-',
            'inherited' if best and origin is not None and best.get('update', 0) <= origin else 'current')


root = Path(sys.argv[1]).expanduser()
print(f"Reading saved histories under: {root}")
paths = sorted(p for p in root.rglob("*.pt") if p.name in ("classification.pt", "detection.pt"))
if not paths:
    print("No saved classification or detection checkpoints yet.")
summary = []
for checkpoint_path in paths:
    try:
        row = show_history(checkpoint_path)
        if row is not None:
            summary.append(row)
    except Exception as error:
        print(f"Could not read {checkpoint_path}: {error}")

print("\nBest validation summary (from saved history; ties use the first occurrence)")
print("| Run / model folder | Model | Metric | Best | Epoch | Update | History |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for row in summary:
    print('| ' + ' | '.join(row) + ' |')

PY
printf '\nHistory report: %s/training_history.txt\n' "$RUNS"
