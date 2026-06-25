#!/usr/bin/env bash
# finetuning/grpo_eval.sh — 评测基线或 RL LoRA
# 用法: bash finetuning/grpo_eval.sh [LORA_DIR] [LIMIT]
#   不传 LORA_DIR → 评基线
set -euo pipefail
CKPT="/cfs/data/private/WangYaoChi/model/joint_ctc_50"
DATA="/cfs/data/private/WangYaoChi/train_data/all/contextasr/train_contextasr2.jsonl"
LORA="${1:-}"
LIMIT="${2:-200}"
if [ -n "$LORA" ]; then
  python -m finetuning.grpo_eval --ckpt "$CKPT" --data "$DATA" --lora "$LORA" --limit "$LIMIT"
else
  python -m finetuning.grpo_eval --ckpt "$CKPT" --data "$DATA" --limit "$LIMIT"
fi
