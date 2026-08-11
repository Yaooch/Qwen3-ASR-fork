#!/usr/bin/env bash
# finetuning/grpo_train.sh — GRPO 训练 wrapper（多卡数据并行）
# 用法:
#   bash finetuning/grpo_train.sh [OUTPUT_DIR] [NPROC] [RESUME]
#   NPROC 默认 8；单卡跑传 1；RESUME=1 时从 OUTPUT_DIR/lora 续训
# effective batch = NPROC × batch_size_per_rank
set -euo pipefail
CKPT="/cfs/data/private/WangYaoChi/model/joint_ctc_50"
DATA="/cfs/data/private/WangYaoChi/train_data/all/contextasr/train_contextasr2_zeroshot_prompt.jsonl"
OUT="${1:-/cfs/data/private/WangYaoChi/model/joint_ctc_50_grpo_4}"
NPROC="${2:-4}"
RESUME="${3:-0}"
RESUME_FROM="${RESUME_FROM:-$OUT/lora}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
PORT="${PORT:-29500}"

torchrun --nproc-per-node="$NPROC" --master-port="$PORT" -m finetuning.grpo_train \
  --ckpt "$CKPT" --data "$DATA" --output_dir "$OUT" \
  --group_size 16 --batch_size_per_rank 8 \
  --temperature 1.0 --lr 5e-5 --beta 0.04 --ppo_epochs "$PPO_EPOCHS" \
  --resume "$RESUME" --resume_from "$RESUME_FROM" \
  --max_steps 5000
