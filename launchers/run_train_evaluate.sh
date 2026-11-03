#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash launchers/run_train_evaluate.sh {mobilenetv2|efficientnet|edgenext}" >&2
  exit 2
fi
FAMILY="$1"
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
RUNS="$(python path.py --get RUNS_DIR)"
RESULTS="$(python path.py --get RESULTS_DIR)"
TRAIN_SUBSET="$(python path.py --get MATCHED_TRAIN_SUBSET)"
[[ -f "$TRAIN_SUBSET" ]] || { printf 'Missing training subset: %s\n' "$TRAIN_SUBSET" >&2; exit 1; }
ROOT="$RUNS/shared_${FAMILY}_subset25_class5_linear_det100_cosine/seed_0"
OUT="$RESULTS/shared_${FAMILY}_subset25_class5_linear_det100_cosine_seed0_$(date +%Y%m%d_%H%M%S)"

# Re-running resumes only this shared-launcher family folder.
printf '\nCheckpoint folder: %s\n' "$ROOT"
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
    --max-val-videos 100 --val-frames-per-video 20 \
    --subset-seed 7 --seed 0 --device cuda

  python -u train_matched.py \
    --experiment "$EXP" --task detection \
    --video-ids "$TRAIN_SUBSET" \
    --checkpoint-out "$DIR/detection.pt" "${DET_RESUME[@]}" \
    --init-checkpoint "$DIR/classification_best.pt" \
    --epochs "$DET_EPOCHS" --lr-schedule cosine --validate-every "$VALIDATE_EVERY" \
    --batch-size "$DET_BATCH" --num-workers "$WORKERS" \
    --clip-length 6 --clip-stride 6 --learning-rate 0.001 \
    --max-val-videos 100 --val-frames-per-video 20 \
    --big-call-interval 8 \
    --subset-seed 7 --seed 0 --device cuda
done

for EXP in $EXPERIMENTS; do
  MODE=big
  case "$EXP" in joint_*) MODE=fixed ;; esac
  printf '\nEvaluating %s\n' "$EXP"
  python -u -m evaluate.evaluate_paired_vod \
    --checkpoint "$ROOT/$EXP/detection_best.pt" \
    --mode "$MODE" --interval 8 \
    --max-videos 100 --frames-per-video 20 \
    --subset-seed 7 --seed 0 \
    --score-threshold 0 --max-detections 100 \
    --warmup-frames 10 --backbone-benchmark-steps 200 --no-losses --no-visuals \
    --device cuda --output-dir "$OUT/$EXP"
done

printf '\nEvaluation results: %s\n' "$OUT"
