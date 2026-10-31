#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON="$(python3 path.py --get PYTHON)"
TRAIN_ANNOTATIONS="$("$PYTHON" path.py --get TRAIN_ANNOTATIONS)"
TRAIN_FRAMES="$("$PYTHON" path.py --get TRAIN_FRAMES)"
VAL_ANNOTATIONS="$("$PYTHON" path.py --get VAL_ANNOTATIONS)"
VAL_FRAMES="$("$PYTHON" path.py --get VAL_FRAMES)"
RESUME_JOINT="$("$PYTHON" path.py --get RESUME_JOINT)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$PYTHON" -u train_paired_vod.py --annotations "$TRAIN_ANNOTATIONS" --frames-root "$TRAIN_FRAMES" --subset-seed 7 --val-annotations "$VAL_ANNOTATIONS" --val-frames-root "$VAL_FRAMES" --val-frames-per-video 20 --batch-size 16 --clip-length 10 --clip-stride 10 --num-workers 2 --prefetch-factor 1 --learning-rate 0.0001 --grad-clip 10.0 --seed 0 --device cuda --validation-interval 4 --fast-init slice-big --fast-width-mult 0.35 --fast-image-size 160 --small-only-state-policy update --fast-bn-momentum 0.01 --reset-fast-bn --fast-learning-rate 0.001 --diagnostic-validation --memory-corruption-probability 0.0 --memory-dropout 0.1 --memory-noise-std 0.1 --freeze-big-in-joint --resume "$RESUME_JOINT" --checkpoint-out "$RESUME_JOINT" --epochs 50 --sharing-strategy file_system --log-every 500
