#!/usr/bin/env bash
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vision
set -u
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

RUNS="$(python path.py --get RUNS_DIR)"
RESULTS="$(python path.py --get RESULTS_DIR)"
ROOT="$(python - "$RUNS" "${1:-}" <<'PY'
import sys
from pathlib import Path
if sys.argv[2]:
    print(Path(sys.argv[2]).expanduser().resolve())
else:
    runs = [p for p in Path(sys.argv[1]).glob('matched_subset25_cosine_100epochs_*') if p.is_dir()]
    if not runs:
        raise SystemExit('No cosine run folder found yet.')
    print(max(runs, key=lambda p: p.stat().st_mtime_ns) / 'seed_0')
PY
)"
CHECKPOINT="$ROOT/joint_120d/detection_best.pt"

[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
mkdir -p "$RESULTS"
OUT="$(mktemp -d "$RESULTS/joint_cosine_interval_analysis_XXXXXX")"
printf '\nSource checkpoint: %s\nResults: %s\n' "$CHECKPOINT" "$OUT"
# Keep all four evaluations on identical weights even while training saves a new best.
cp "$CHECKPOINT" "$OUT/joint_best.pt"
CHECKPOINT="$OUT/joint_best.pt"

for SETTING in big_only interval_2 interval_4 interval_8; do
  MODE=big
  INTERVAL=8
  case "$SETTING" in
    interval_*) MODE=fixed; INTERVAL="${SETTING#interval_}" ;;
  esac
  printf '\nEvaluating joint: %s\n' "$SETTING"
  python -u -m evaluate.evaluate_paired_vod \
    --checkpoint "$CHECKPOINT" \
    --mode "$MODE" --interval "$INTERVAL" \
    --max-videos 100 --frames-per-video 20 \
    --subset-seed 7 --seed 0 \
    --score-threshold 0 --max-detections 100 \
    --warmup-frames 10 --no-losses --no-visuals \
    --device cuda --output-dir "$OUT/$SETTING"
done

python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
files = {
    "Joint big-only": out / "big_only/big/metrics.json",
    "Joint k=2": out / "interval_2/fixed_k2/metrics.json",
    "Joint k=4": out / "interval_4/fixed_k4/metrics.json",
    "Joint k=8": out / "interval_8/fixed_k8/metrics.json",
}
results = {name: json.loads(path.read_text()) for name, path in files.items()}
reference = results["Joint big-only"]
checks = (
    "evaluated_image_ids", "positive_counts_per_class", "AP_definition",
    "score_threshold", "nms_iou_threshold", "max_detections_per_class",
    "max_total_detections", "detector_fingerprint", "class_names", "video_ids",
)
for name, result in results.items():
    for key in checks:
        if result[key] != reference[key]:
            raise SystemExit(f"Cannot compare: {name} differs in {key}")

names = reference["class_names"]
lines = [
    "AP values are percentages. — means no ground-truth examples.", "",
    "| Metric / class | " + " | ".join(results) + " |",
    "|---|" + "---:|" * len(results),
]
rows = [("mAP@0.5", [r["mAP@0.5"] for r in results.values()])]
for category_id in sorted(names, key=int):
    rows.append((f"{category_id}: {names[category_id]}".replace("|", "/"),
                 [r["AP@0.5_per_class"].get(category_id) for r in results.values()]))
for label, values in rows:
    cells = ["—" if value is None else f"{100 * value:.2f}" for value in values]
    lines.append("| " + label + " | " + " | ".join(cells) + " |")
report = "\n".join(lines)
(out / "comparison.md").write_text(report + "\n")
print(f"\nVerified identical {len(reference['evaluated_image_ids'])} validation frames.")
print(report)
print(f"\nSaved comparison: {out / 'comparison.md'}")
PY
