#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniforge3/bin/activate"
conda activate autotrain

cd "$(dirname "$0")/.."

autotrain llm \
  --train \
  --model codellama/CodeLlama-7b-Instruct-hf \
  --project-name deltamcp-codellama7b-lora \
  --data-path DeltaMCP/llm-finetuned/autotrain \
  --train-split train \
  --valid-split valid \
  --text-column text \
  --epochs 3 \
  --batch-size 2 \
  --gradient-accumulation 2 \
  --lr 0.00015 \
  --warmup-ratio 0.1 \
  --optimizer adamw_bnb_8bit \
  --scheduler cosine \
  --weight-decay 0.01 \
  --max-grad-norm 0.3 \
  --peft \
  --lora-r 16 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --max-length 2048 \
  --block-size 1536 \
  --logging-steps 10 \
  --evaluation-strategy steps \
  --save-strategy steps \
  --save-total-limit 2 \
  --mixed-precision bf16 \
  --quantization int4 \
  --seed 42
