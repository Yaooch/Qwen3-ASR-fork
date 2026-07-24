#!/bin/bash
# finetuning/train_nlu.sh — NLU(用户意图提取)LoRA SFT wrapper
# 用法: bash finetuning/train_nlu.sh [GPU_IDS]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

gpu_ids="0,1,2,3,4,5,6,7"
if [[ $# -gt 0 ]]; then
    gpu_ids="$1"
fi
num_gpus=$(echo "$gpu_ids" | awk -F',' '{print NF}')

export CUDA_VISIBLE_DEVICES=$gpu_ids
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 基线 joint checkpoint(已有 ASR 能力)
ckpt="/cfs/data/private/WangYaoChi/model/joint_ctc_50"
# ckpt="/cfs/data/private/hubk/Qwen3-ASR/Qwen/Qwen3-ASR-1___7B"
train_file="/root/asr_project/Qwen3-ASR/voyah_agent_train_split.jsonl"
# train_file="/root/asr_project/Qwen3-ASR/all_full_nlu_train.jsonl"
output_dir="/cfs/data/private/WangYaoChi/model/joint_ctc_50_nlu_9"
logging_dir="./logs/logs_ctc_50_nlu_9"

batch_size=4
grad_acc=16
epochs=10
lr=1e-5
lora_r=32
lora_alpha=64
save_steps=500
num_workers=4
master_port=$(shuf -n 1 -i 20000-65000)

# 全参微调: FULL_FT=1 bash train_nlu.sh (lr 建议降到 1e-5, batch 视显存调小)
full_ft="${FULL_FT:-1}"
full_ft_flag=()
if [[ "$full_ft" == "1" ]]; then
    full_ft_flag=(--full_ft)
fi

echo "==========================================================="
echo "  启动 Qwen3-ASR NLU LoRA SFT"
echo "  GPU: $gpu_ids  数量: $num_gpus"
echo "  基线: $ckpt"
echo "  输出: $output_dir"
echo "==========================================================="

torchrun --nproc_per_node="$num_gpus" --master_port "$master_port" \
    train_nlu.py \
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
    --max_len 2048 \
    --logging_dir "$logging_dir" \
    --num_workers "$num_workers" \
    "${full_ft_flag[@]}"
