#!/bin/bash
# qwen_asr_ext/nlu/scripts/train.sh — joint / 纯 Qwen3 文本 NLU SFT
# 用法: BACKEND=joint|llm FULL_FT=0|1 bash qwen_asr_ext/nlu/scripts/train.sh [GPU_IDS]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

backend="${BACKEND:-joint}"
case "$backend" in
    joint)
        default_gpus="0,1,2,3,4,5,6,7"
        ckpt="/cfs/data/private/WangYaoChi/model/joint_ctc_50"
        train_file="/cfs/data/private/WangYaoChi/train_data/all/nlu/voyah_agent_train_split.jsonl"
        output_dir="/cfs/data/private/WangYaoChi/model/joint_ctc_50_nlu_9"
        logging_dir="./logs/logs_ctc_50_nlu_9"
        batch_size=4
        grad_acc=16
        epochs=10
        lr=1e-5
        save_steps=500
        ;;
    llm)
        default_gpus="6,7"
        ckpt="/cfs/data/private/WangYaoChi/model/Qwen3-1.7B"
        train_file="/cfs/data/private/WangYaoChi/train_data/all/nlu/voyah_agent_train.jsonl"
        output_dir="/cfs/data/private/WangYaoChi/model/qwen3_1_7b_nlu_6"
        logging_dir="./logs/logs_qwen3_nlu_6"
        batch_size=4
        grad_acc=4
        epochs=2
        lr=2e-5
        save_steps=1000
        ;;
    *)
        echo "BACKEND 只支持 joint/llm"
        exit 2
        ;;
esac

gpu_ids="${1:-$default_gpus}"
num_gpus="$(awk -F',' '{print NF}' <<< "$gpu_ids")"
export CUDA_VISIBLE_DEVICES="$gpu_ids"
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

full_ft="${FULL_FT:-1}"
full_ft_flag=()
if [[ "$full_ft" == "1" ]]; then
    full_ft_flag=(--full_ft)
fi

echo "NLU SFT: backend=$backend gpu=$gpu_ids full_ft=$full_ft output=$output_dir"
torchrun --nproc_per_node="$num_gpus" --master_port "$(shuf -n 1 -i 20000-65000)" \
    -m qwen_asr_ext.nlu.train \
    --backend "$backend" \
    --ckpt "$ckpt" \
    --train_file "$train_file" \
    --output_dir "$output_dir" \
    --batch_size "$batch_size" \
    --grad_acc "$grad_acc" \
    --epochs "$epochs" \
    --lr "$lr" \
    --lora_r 32 \
    --lora_alpha 64 \
    --save_steps "$save_steps" \
    --max_len 2048 \
    --logging_dir "$logging_dir" \
    --num_workers 4 \
    "${full_ft_flag[@]}"
