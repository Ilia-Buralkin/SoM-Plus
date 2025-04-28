#!/usr/bin/env bash
set -euo pipefail

# Fixed parameters
MODEL="gemini-2.0-flash"
QPS=1.0

# The script’s built‑in default for --output_dir will be used
# Looped options
SEGMENTATIONS=(sam2)
METHODS=(parallel unified baseline)

for SEG in "${SEGMENTATIONS[@]}"; do
  for METH in "${METHODS[@]}"; do
    echo ">>> Running refcocog_test.py with model=$MODEL, segmentation=$SEG, method=$METH"
    python refcocog_test.py \
      --model        "$MODEL" \
      --segmentation "$SEG" \
      --method       "$METH" \
      --qps          "$QPS"
  done
done
