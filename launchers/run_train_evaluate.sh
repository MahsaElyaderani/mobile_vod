#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash launchers/run_train_evaluate.sh {mobilenetv2|efficientnet|edgenext} [all|train|eval]" >&2
  exit 2
fi
FAMILY="$1"
ACTION="${2:-all}"
case "$ACTION" in
  all|train|eval) ;;
  *) echo "Unknown mode: $ACTION" >&2; exit 2 ;;
esac
case "$FAMILY" in
  mobilenetv2) EXPERIMENTS="small_050 big_120d joint_120d" ;;
  efficientnet) EXPERIMENTS="small_tinynet_d big_efficientnet_b0 joint_efficientnet_b0_tinynet_d" ;;
  edgenext) EXPERIMENTS="small_edgenext_xx big_edgenext_small joint_edgenext" ;;
  *) echo "Unknown family: $FAMILY" >&2; exit 2 ;;
esac

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vision
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Matched settings for all three backbone families.
CLASS_EPOCHS=5
DET_EPOCHS=100
CLASS_BATCH=8
DET_BATCH=4
WORKERS=4
VALIDATE_EVERY=5000
# Override these with environment variables, or edit the defaults here.
MAX_VAL_VIDEOS="${MAX_VAL_VIDEOS:-100}"
VAL_FRAMES_PER_VIDEO="${VAL_FRAMES_PER_VIDEO:-20}"
SUBSET_SEED="${SUBSET_SEED:-7}"
RUNS="$(python path.py --get RUNS_DIR)"
RESULTS="$(python path.py --get RESULTS_DIR)"
TRAIN_SUBSET="${TRAIN_SUBSET:-0.25}"
case "$TRAIN_SUBSET" in
  0.25) TRAIN_SUBSET="$PWD/splits/train_25pct.json" ;;
  0.5) TRAIN_SUBSET="$PWD/splits/train_50pct.json" ;;
esac
SUBSET_TAG="$(basename "$TRAIN_SUBSET" .json)"
case "$SUBSET_TAG" in
  train_10pct|train_25pct|train_50pct) SUBSET_TAG="subset${SUBSET_TAG#train_}"; SUBSET_TAG="${SUBSET_TAG%pct}" ;;
esac
DEFAULT_RUN="shared_${FAMILY}_${SUBSET_TAG}_class${CLASS_EPOCHS}_linear_det${DET_EPOCHS}_cosine"
if [[ "$MAX_VAL_VIDEOS/$VAL_FRAMES_PER_VIDEO/$SUBSET_SEED" != "100/20/7" ]]; then
  DEFAULT_RUN="${DEFAULT_RUN}_val${MAX_VAL_VIDEOS}x${VAL_FRAMES_PER_VIDEO}_seed${SUBSET_SEED}"
fi
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN}"
ROOT="$RUNS/$RUN_NAME/seed_0"

# Re-running resumes only this shared-launcher family folder.
printf '\nCheckpoint folder: %s\n' "$ROOT"
if [[ "$ACTION" != eval ]]; then
[[ -f "$TRAIN_SUBSET" ]] || { printf 'Missing training subset: %s\n' "$TRAIN_SUBSET" >&2; exit 1; }
for EXP in $EXPERIMENTS; do
  DIR="$ROOT/$EXP"
  CLASS_RESUME=()
  DET_RESUME=()
  if [[ -f "$DIR/classification.pt" ]]; then CLASS_RESUME=(--resume "$DIR/classification.pt"); fi
  if [[ -f "$DIR/detection.pt" ]]; then DET_RESUME=(--resume "$DIR/detection.pt"); fi
  printf '\nTraining %s\n' "$EXP"
  python -u train_matched.py \
    --experiment "$EXP" --task classification \
    --video-ids "$TRAIN_SUBSET" \
    --checkpoint-out "$DIR/classification.pt" "${CLASS_RESUME[@]}" \
    --epochs "$CLASS_EPOCHS" --lr-schedule linear --validate-every "$VALIDATE_EVERY" \
    --batch-size "$CLASS_BATCH" --num-workers "$WORKERS" \
    --steps 3 --learning-rate 0.01 \
    --max-val-videos "$MAX_VAL_VIDEOS" --val-frames-per-video "$VAL_FRAMES_PER_VIDEO" \
    --subset-seed "$SUBSET_SEED" --seed 0 --device cuda

  python -u train_matched.py \
    --experiment "$EXP" --task detection \
    --video-ids "$TRAIN_SUBSET" \
    --checkpoint-out "$DIR/detection.pt" "${DET_RESUME[@]}" \
    --init-checkpoint "$DIR/classification_best.pt" \
    --epochs "$DET_EPOCHS" --lr-schedule cosine --validate-every "$VALIDATE_EVERY" \
    --batch-size "$DET_BATCH" --num-workers "$WORKERS" \
    --clip-length 6 --clip-stride 6 --learning-rate 0.001 \
    --max-val-videos "$MAX_VAL_VIDEOS" --val-frames-per-video "$VAL_FRAMES_PER_VIDEO" \
    --big-call-interval 8 \
    --subset-seed "$SUBSET_SEED" --seed 0 --device cuda
done
fi

if [[ "$ACTION" != train ]]; then
for EXP in $EXPERIMENTS; do
  [[ -f "$ROOT/$EXP/detection_best.pt" ]] || {
    echo "Missing best detection checkpoint: $ROOT/$EXP/detection_best.pt" >&2; exit 1;
  }
done
mkdir -p "$RESULTS"
OUT="$(mktemp -d "$RESULTS/${RUN_NAME}_seed0_XXXXXX")"
for EXP in $EXPERIMENTS; do
  MODE=big
  case "$EXP" in joint_*) MODE=fixed ;; esac
  printf '\nEvaluating %s\n' "$EXP"
  python -u -m evaluate.evaluate_paired_vod \
    --checkpoint "$ROOT/$EXP/detection_best.pt" \
    --mode "$MODE" --interval 8 \
    --max-videos "$MAX_VAL_VIDEOS" --frames-per-video "$VAL_FRAMES_PER_VIDEO" \
    --subset-seed "$SUBSET_SEED" --seed 0 \
    --score-threshold 0 --max-detections 100 \
    --warmup-frames 10 --backbone-benchmark-steps 200 --no-losses --no-visuals \
    --device cuda --output-dir "$OUT/$EXP"
done

printf '\nEvaluation results: %s\n' "$OUT"
fi
