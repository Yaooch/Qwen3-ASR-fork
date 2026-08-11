#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

gpu_ids="0,1,2,3"
if [[ $# -gt 0 ]]; then
    gpu_ids="$1"
fi
num_gpus=$(echo "$gpu_ids" | awk -F',' '{print NF}')

echo "==========================================================="
echo "  启动 Qwen3-ASR 联合微调"
echo "  指定使用显卡 ID : $gpu_ids"
echo "  检测到 GPU 数量 : $num_gpus"
echo "==========================================================="

export CUDA_VISIBLE_DEVICES=$gpu_ids
export OMP_NUM_THREADS=4

model_path="/cfs/data/private/WangYaoChi/model/joint_ctc_50"
# model_path="/cfs/data/private/hubk/Qwen3-ASR/Qwen/Qwen3-ASR-1___7B"
train_file="/cfs/data/private/WangYaoChi/train_data/all/contextasr/train_contextasr2.jsonl"
eval_file="/cfs/data/private/WangYaoChi/train_data/all/contextasr/eval_contextasr.jsonl"
output_dir="/cfs/data/private/WangYaoChi/model/joint_ctc_50_sft"
# output_dir="/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14-hotword-2"
logging_dir="./logs/logs_ctc_50_sft"

batch_size=32
grad_acc=4
epochs=1

# encoder, proj, llm, ctc, rnnt 可以设置不同的学习率，或者只微调其中部分模块
train_tasks="llm"

lr_encoder=4e-5
lr_llm=1e-5
lr_proj=4e-5
lr_ctc=2e-3
lr_rnnt=2e-3

w_llm=1
w_ctc=0
w_rnnt=0

# CTC adapter: auto 继承源 checkpoint；新 MoE 训练可设为 moe
ctc_adapter="mlp"

stream_train=1

save_steps=200
num_workers=4
master_port=$(shuf -n 1 -i 20000-65000)

if [[ -z "$train_file" ]]; then
    echo "请先在 train.sh 中设置 train_file"
    exit 1
fi

torchrun \
    --nproc_per_node="$num_gpus" \
    --master_port="$master_port" \
    -m qwen_asr_ext.joint.train \
    --model_path "$model_path" \
    --train_file "$train_file" \
    --eval_file "$eval_file" \
    --resume 1 \
    --output_dir "$output_dir" \
    --batch_size "$batch_size" \
    --grad_acc "$grad_acc" \
    --train "$train_tasks" \
    --lr_llm "$lr_llm" \
    --lr_proj "$lr_proj" \
    --lr_encoder "$lr_encoder" \
    --lr_ctc "$lr_ctc" \
    --lr_rnnt "$lr_rnnt" \
    --w_llm "$w_llm" \
    --w_ctc "$w_ctc" \
    --w_rnnt "$w_rnnt" \
    --ctc_adapter "$ctc_adapter" \
    --epochs "$epochs" \
    --stream_train "$stream_train" \
    --save_steps "$save_steps" \
    --log_steps 10 \
    --logging_dir "$logging_dir" \
    --num_workers "$num_workers" \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \