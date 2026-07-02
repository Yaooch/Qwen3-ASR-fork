#!/bin/bash
# finetuning/train_asr_nlu.sh — ASR + ASR+NLU 联合 LoRA SFT wrapper
# 用法: bash finetuning/train_asr_nlu.sh [GPU_IDS]
#
# 用同一批 TTS 两用数据派生 ASR + ASR+NLU 两种样本联合训练，防止只训 NLU 导致 ASR 崩。
# 数据每行 {"messages":[{system},{user:audio_path},{assistant="language X<asr_text>文本\n意图JSON"}]}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

gpu_ids="0,1,2,3"
if [[ $# -gt 0 ]]; then
    gpu_ids="$1"
fi
num_gpus=$(echo "$gpu_ids" | awk -F',' '{print NF}')

export CUDA_VISIBLE_DEVICES=$gpu_ids
export OMP_NUM_THREADS=4

ckpt="/cfs/data/private/WangYaoChi/model/joint_ctc_50"
train_file="/cfs/data/private/WangYaoChi/train_data/all/nlu/voyah_asr_nlu_2.jsonl"
output_dir="/cfs/data/private/WangYaoChi/model/joint_ctc_50_asr_nlu_1"
logging_dir="./logs/logs_asr_nlu_1"

batch_size=16
grad_acc=4
epochs=3
lr=1e-4
lora_r=32
lora_alpha=64
save_steps=500
num_workers=4
master_port=$(shuf -n 1 -i 20000-65000)

echo "==========================================================="
echo "  启动 Qwen3-ASR ASR+NLU 联合 LoRA SFT"
echo "  GPU: $gpu_ids  数量: $num_gpus"
echo "  基线: $ckpt"
echo "  数据: $train_file (每条派生 ASR + ASR+NLU)"
echo "  输出: $output_dir"
echo "==========================================================="

torchrun --nproc_per_node="$num_gpus" --master_port "$master_port" \
    train_asr_nlu.py \
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
    --num_workers "$num_workers"
