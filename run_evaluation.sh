(
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
PYTHON="$(python3 path.py --get PYTHON)"
BIG="$("$PYTHON" path.py --get EVALUATION_BIG)"
SMALL="$("$PYTHON" path.py --get EVALUATION_SMALL)"
JOINT="$("$PYTHON" path.py --get EVALUATION_JOINT)"
ANNOTATIONS="$("$PYTHON" path.py --get VAL_ANNOTATIONS)"
FRAMES="$("$PYTHON" path.py --get VAL_FRAMES)"
RESULTS="$("$PYTHON" path.py --get RESULTS_DIR)"

# Check all inputs before starting.
for CHECKPOINT in "$BIG" "$SMALL" "$JOINT"; do
  if [[ ! -f "$CHECKPOINT" ]]; then
    printf 'Checkpoint not found: %s\n' "$CHECKPOINT" >&2
    exit 1
  fi
done

OUTPUT="$RESULTS/standalone_vs_joint_20frames_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$OUTPUT"

# Evaluate fixed snapshots rather than changing training files.
#cp -- "$BIG" "$OUTPUT/big_snapshot.pt"
#cp -- "$SMALL" "$OUTPUT/small_snapshot.pt"
#cp -- "$JOINT" "$OUTPUT/joint_snapshot.pt"

COMMON=(
  --annotations "$ANNOTATIONS"
  --frames-root "$FRAMES"
  --memory-feedback clean
  --frames-per-video 20
  --subset-seed 7
  --seed 0
  --device "${DEVICE:-cuda}"
  --score-threshold 0
  --max-detections 100
  --warmup-frames 10
  --no-losses
  --no-visuals
)
# Standalone alpha=1.0: its own backbone, LSTM, and SSD.
"$PYTHON" -u -m evaluate.evaluate_paired_vod "${COMMON[@]}" \
  --checkpoint "$BIG" \
  --native-big --mode big \
  --output-dir "$OUTPUT/standalone_big"
# Standalone alpha=0.35: its own backbone, LSTM, and SSD.
"$PYTHON" -u -m evaluate.evaluate_paired_vod "${COMMON[@]}" \
  --checkpoint "$SMALL" \
  --native-big --mode big \
  --output-dir "$OUTPUT/standalone_small"
# Joint model at matching validation frames.
for K in 4 8 16; do
  "$PYTHON" -u -m evaluate.evaluate_paired_vod "${COMMON[@]}" \
    --checkpoint "$JOINT" \
    --mode fixed --interval "$K" \
    --output-dir "$OUTPUT/joint_k$K"
done

printf '\nResults saved to: %s\n' "$OUTPUT"
)
