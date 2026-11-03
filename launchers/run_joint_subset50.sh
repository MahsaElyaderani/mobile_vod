#!/usr/bin/env bash
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vision
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RUNS="$(python path.py --get RUNS_DIR)"
# Optional first argument overrides the source best checkpoint for the first run.
BEST="${1:-$RUNS/matched_subset25_cosine_100epochs_9fjlhg/seed_0/joint_120d/detection_best.pt}"
OUT="$RUNS/joint_120d_subset50_cosine_continued/detection.pt"
SUBSET="splits/train_50pct.json"
[[ -f "$SUBSET" ]] || { echo "Missing subset: $SUBSET" >&2; exit 1; }

if [[ -f "$OUT" ]]; then
  RESUME=(--resume "$OUT")
  printf 'Resuming latest checkpoint: %s\n' "$OUT"
else
  [[ -f "$BEST" ]] || { echo "Missing source checkpoint: $BEST" >&2; exit 1; }
  RESUME=(--resume "$BEST" --expand-subset)
  printf 'Continuing from best checkpoint: %s\n' "$BEST"
fi

python -u train_matched.py \
  --experiment joint_120d --task detection \
  "${RESUME[@]}" \
  --video-ids "$SUBSET" --checkpoint-out "$OUT" \
  --num-workers 4 --device cuda
