#!/usr/bin/env bash
set -e

MODEL_PATH="/Volumes/KIOXIA/AI Models/lmstudio-community/gemma-3-270m-it-MLX-8bit"
DATA_DIR="./data_ecg_clinical"
ADAPTER_DIR="./adapters_ecg_clinical"
ITERS=${1:-150}

echo "============================================================"
echo "Gemma 3 270M Electrophysiology Reporting LoRA Fine-Tuning"
echo "Base Model: $MODEL_PATH"
echo "Dataset:    $DATA_DIR"
echo "Iterations: $ITERS"
echo "============================================================"

uv run --with mlx-lm mlx_lm.lora \
  --model "$MODEL_PATH" \
  --data "$DATA_DIR" \
  --train \
  --iters "$ITERS" \
  --batch-size 2 \
  --num-layers 8 \
  --learning-rate 2e-4 \
  --adapter-path "$ADAPTER_DIR" \
  --save-every 50 \
  --val-batches 4 \
  --mask-prompt

echo ""
echo "LoRA fine-tuning completed! Adapter saved to '$ADAPTER_DIR'."
