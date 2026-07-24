#!/bin/bash
# finetuning/train_nlu_pure.sh — 纯 Qwen3 LLM NLU SFT wrapper
# 用原始 Qwen3-1.7B（未经 ASR 训练）做 NLU 微调
# 用法: bash finetuning/train_nlu_pure.sh [GPU_IDS]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

gpu_ids="6,7"
if [[ $# -gt 0 ]]; then
    gpu_ids="$1"
fi
num_gpus=$(echo "$gpu_ids" | awk -F',' '{print NF}')

export CUDA_VISIBLE_DEVICES=$gpu_ids
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 原始 Qwen3-1.7B（未经 ASR 训练的纯 LLM）
ckpt="/cfs/data/private/WangYaoChi/model/Qwen3-1.7B"
train_file="/cfs/data/share/NLU/llm_training_data/LLM/20260716/generate/voyah_agent_train.jsonl"
output_dir="/cfs/data/private/WangYaoChi/model/qwen3_1_7b_nlu_6"
logging_dir="./logs/logs_qwen3_nlu_6"

batch_size=4
grad_acc=4
epochs=2
lr=2e-5
lora_r=32
lora_alpha=64
save_steps=1000
num_workers=4
master_port=$(shuf -n 1 -i 20000-65000)

# 全参微调: FULL_FT=1 bash train_nlu_pure.sh
# LoRA 微调: FULL_FT=0 bash train_nlu_pure.sh
full_ft="${FULL_FT:-1}"
full_ft_flag=()
if [[ "$full_ft" == "1" ]]; then
    full_ft_flag=(--full_ft)
fi

echo "==========================================================="
echo "  纯 Qwen3-1.7B NLU SFT"
echo "  GPU: $gpu_ids  数量: $num_gpus"
echo "  基线: $ckpt"
echo "  模式: $([[ "$full_ft" == "1" ]] && echo '全参微调' || echo 'LoRA')"
echo "  输出: $output_dir"
echo "==========================================================="

torchrun --nproc_per_node="$num_gpus" --master_port "$master_port" \
    train_nlu_pure.py \
    --ckpt "$ckpt" \
    --train_file "$train_file" \
    --output_dir "$output_dir" \
    --batch_size "$batch_size" \
    --grad_acc "$grad_acc" \
    --epochs "$epochs" \
    --lr "$lr" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_alpha" \
    --save_steps "$save_steps" \
    --logging_dir "$logging_dir" \
    --num_workers "$num_workers" \
    "${full_ft_flag[@]}"
