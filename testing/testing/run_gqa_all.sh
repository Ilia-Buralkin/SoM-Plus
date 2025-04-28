#!/usr/bin/env bash
set -euo pipefail

# Fixed parameters
MODEL="gemini-1.5-pro"

# Looped options
SEGMENTATIONS=(maskformer sam2)
METHODS=(unified som_baseline parallel)

for SEG in "${SEGMENTATIONS[@]}"; do
  for METH in "${METHODS[@]}"; do

    echo ">>> Running: model=$MODEL, segmentation=$SEG, method=$METH"
    python gqa_test.py \
      --model    "$MODEL" \
      --segmentation "$SEG" \
      --method   "$METH" 
  done
done
