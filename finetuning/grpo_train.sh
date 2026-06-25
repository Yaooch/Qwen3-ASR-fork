#!/usr/bin/env bash
# finetuning/grpo_train.sh — GRPO 训练 wrapper
# 用法: bash finetuning/grpo_train.sh [OUTPUT_DIR]
set -euo pipefail
CKPT="/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-1/checkpoint-228"
DATA="/cfs/data/private/WangYaoChi/train_data/all/contextasr/train_contextasr2.jsonl"
OUT="${1:-/cfs/data/private/WangYaoChi/model/grpo_lora_out}"
python -m finetuning.grpo_train \
  --ckpt "$CKPT" --data "$DATA" --output_dir "$OUT" \
  --group_size 8 --temperature 0.8 --lr 1e-5 --beta 0.04 \
  --max_steps 1000
