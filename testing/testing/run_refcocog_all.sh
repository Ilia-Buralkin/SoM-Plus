#!/usr/bin/env bash
set -euo pipefail

# Fixed parameters
MODEL="gemini-2.0-flash"

# The script’s built‑in default for --output_dir will be used
# Looped options
SEGMENTATIONS=(sam2 maskformer)
METHODS=(parallel unified)

for SEG in "${SEGMENTATIONS[@]}"; do
  for METH in "${METHODS[@]}"; do
    echo ">>> Running refcocog_test.py with model=$MODEL, segmentation=$SEG, method=$METH"
    python refcocog_test_vertex.py \
      --model        "$MODEL" \
      --segmentation "$SEG" \
      --method       "$METH" 
  done
done
