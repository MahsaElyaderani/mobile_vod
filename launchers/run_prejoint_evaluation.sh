#!/usr/bin/env bash
# Run AFTER training, with the GPU otherwise idle. Never launches training.
set -euo pipefail
if [[ $# -ne 0 && $# -ne 2 ]]; then
  echo "Usage: bash launchers/run_prejoint_evaluation.sh [checkpoint output_directory]; defaults are in path.py" >&2
  exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="$(python3 path.py --get PYTHON)"
PAIR_DIR="$("$PYTHON_BIN" path.py --get PREJOINT_PAIR_DIR)"
ANNOTATIONS="$("$PYTHON_BIN" path.py --get VAL_ANNOTATIONS)"
FRAMES="$("$PYTHON_BIN" path.py --get VAL_FRAMES)"
if [[ $# -eq 2 ]]; then
  BIG_CHECKPOINT="$1"
  OUTPUT="$2"
else
  BIG_CHECKPOINT="$("$PYTHON_BIN" path.py --get PREJOINT_BIG)"
  OUTPUT="$("$PYTHON_BIN" path.py --get PREJOINT_OUTPUT)"
fi
for checkpoint in "$BIG_CHECKPOINT" "$PAIR_DIR/detector_big_reference.pt" \
  "$PAIR_DIR/detector_small_zero_memory.pt" "$PAIR_DIR/detector_small_pre_joint.pt"; do
  [[ -f "$checkpoint" ]] || { echo "Required checkpoint missing: $checkpoint" >&2; exit 1; }
done
[[ ! -e "$OUTPUT" ]] || { echo "Use a fresh output directory: $OUTPUT" >&2; exit 1; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
COMMON=(--annotations "$ANNOTATIONS"
  --frames-root "$FRAMES" --subset-seed 7 --seed 0
  --score-threshold 0 --max-detections 100 --warmup-frames 3
  --device "${DEVICE:-cuda}" --no-visuals --no-losses)
# Set FRAMES_PER_VIDEO=full for full-video validation; keep these results separate.
if [[ "${FRAMES_PER_VIDEO:-20}" != full ]]; then
  COMMON+=(--frames-per-video "${FRAMES_PER_VIDEO:-20}")
fi
mkdir -p -- "$OUTPUT"
evaluate() {
  local label="$1" checkpoint="$2" mode="$3" feedback="$4"
  shift 4
  "$PYTHON_BIN" -u -m evaluate.evaluate_paired_vod "${COMMON[@]}" \
    --checkpoint "$checkpoint" --mode "$mode" --memory-feedback "$feedback" \
    --output-dir "$OUTPUT/$label" "$@"
}
evaluate big_independent "$BIG_CHECKPOINT" big clean --native-big
evaluate big_reference "$PAIR_DIR/detector_big_reference.pt" big clean
evaluate small_e10_zero "$PAIR_DIR/detector_small_zero_memory.pt" small none
evaluate fixed4_e10 "$PAIR_DIR/detector_small_zero_memory.pt" fixed clean --interval 4
evaluate small_e15_zero "$PAIR_DIR/detector_small_pre_joint.pt" small none
evaluate small_e15_self "$PAIR_DIR/detector_small_pre_joint.pt" small clean
evaluate fixed4_e15 "$PAIR_DIR/detector_small_pre_joint.pt" fixed clean --interval 4
evaluate fixed4_e15_no_memory "$PAIR_DIR/detector_small_pre_joint.pt" fixed none --interval 4
evaluate fixed8_e15 "$PAIR_DIR/detector_small_pre_joint.pt" fixed clean --interval 8
evaluate fixed16_e15 "$PAIR_DIR/detector_small_pre_joint.pt" fixed clean --interval 16
"$PYTHON_BIN" - "$OUTPUT" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
with (root / 'all_metrics.csv').open('x', newline='') as stream:
    table = csv.writer(stream)
    table.writerow(['experiment', 'AP50_percent', 'FPS', 'conv_linear_GFLOPs_per_frame',
                    'big_call_frequency', 'checkpoint_sha256', 'epoch'])
    for path in sorted(root.glob('*/*/metrics.json')):
        item = json.loads(path.read_text())
        source = item['evaluation_source']
        table.writerow([path.parent.parent.name, 100 * item['mAP@0.5'], item['FPS'],
                        item['conv_linear_FLOPs_per_frame'] / 1e9, item['big_call_frequency'],
                        source['checkpoint_sha256'], source['epoch']])
PY
"$PYTHON_BIN" -m evaluate.compare_paired_runs --frames-root "$FRAMES" \
  --runs "$OUTPUT/big_independent/big" "$OUTPUT/big_reference/big" \
    "$OUTPUT/small_e10_zero/small" "$OUTPUT/small_e15_zero/small" \
    "$OUTPUT/fixed4_e10/fixed_k4" "$OUTPUT/fixed4_e15/fixed_k4" \
  --labels big_independent big_reference small_e10_zero small_e15_zero fixed4_e10 fixed4_e15 \
  --max-side 320 --save-frames-every 10 --output-dir "$OUTPUT/comparison"
