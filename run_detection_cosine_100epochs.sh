#!/usr/bin/env bash
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vision
set -u
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

EXPERIMENTS="joint_120d big_120d small_050"
RUNS="$(python path.py --get RUNS_DIR)"
TRAIN_SUBSET="$(python path.py --get MATCHED_TRAIN_SUBSET)"
SOURCE="$RUNS/matched_subset25_linear_1epoch/seed_0"
[[ -f "$TRAIN_SUBSET" ]] || { echo "Missing subset: $TRAIN_SUBSET" >&2; exit 1; }
for EXP in $EXPERIMENTS; do
  [[ -f "$SOURCE/$EXP/classification_best.pt" ]] || {
    echo "Missing classification checkpoint: $SOURCE/$EXP/classification_best.pt" >&2
    exit 1
  }
done

# Every invocation starts a fresh experiment; previous runs are read-only inputs.
ROOT="$(mktemp -d "$RUNS/matched_subset25_cosine_100epochs_XXXXXX")"
printf '\nNew detection runs: %s/seed_0\n' "$ROOT"
for EXP in $EXPERIMENTS; do
  printf '\nTraining %s detection\n' "$EXP"
  python -u train_matched.py \
    --experiment "$EXP" --task detection \
    --video-ids "$TRAIN_SUBSET" \
    --init-checkpoint "$SOURCE/$EXP/classification_best.pt" \
    --checkpoint-out "$ROOT/seed_0/$EXP/detection.pt" \
    --epochs 100 --lr-schedule cosine --learning-rate 0.001 \
    --batch-size 4 --num-workers 4 \
    --clip-length 6 --clip-stride 6 \
    --validate-every 5000 --max-val-videos 100 --val-frames-per-video 20 \
    --big-call-interval 8 --subset-seed 7 --seed 0 --device cuda
done

printf '\nDetection checkpoints: %s/seed_0\n' "$ROOT"
