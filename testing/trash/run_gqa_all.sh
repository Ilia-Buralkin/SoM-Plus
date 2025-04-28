#!/usr/bin/env bash
set -euo pipefail

# Fixed parameters
MODEL="gemini-2.0-flash"
QPS=1.0

# Looped options
SEGMENTATIONS=(maskformer sam2)
METHODS=(baseline parallel unified_pro)

for SEG in "${SEGMENTATIONS[@]}"; do
  for METH in "${METHODS[@]}"; do

    echo ">>> Running: model=$MODEL, segmentation=$SEG, method=$METH"
    python gqa_test.py \
      --model    "$MODEL" \
      --segmentation "$SEG" \
      --method   "$METH" \
      --qps      "$QPS"
  done
done
