#!/bin/bash
# finetuning/infer_nlu.sh — joint / 纯 Qwen3 文本 NLU 推理评测
# 用法: BACKEND=joint|llm TASK=nlu|agent EVAL=0|1 bash finetuning/infer_nlu.sh [INPUT] [GPU_IDS]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

backend="${BACKEND:-joint}"
case "$backend" in
    joint)
        default_input="/cfs/data/private/WangYaoChi/train_data/all/nlu/voyah_agent_train_split_3k.jsonl"
        default_gpus="0,1,2,3"
        ckpt="/cfs/data/private/WangYaoChi/model/joint_ctc_50_nlu_9"
        output_dir="/cfs/data/private/WangYaoChi/test_out/joint_ctc_50_nlu_9_train"
        ;;
    llm)
        default_input="/cfs/data/private/WangYaoChi/train_data/all/nlu/voyah_agent_train_20260716_sample_2k.jsonl"
        default_gpus="6,7"
        ckpt="/cfs/data/private/WangYaoChi/model/qwen3_1_7b_nlu_6"
        output_dir="/cfs/data/private/WangYaoChi/test_out/qwen3_1_7b_nlu_6"
        ;;
    *)
        echo "BACKEND 只支持 joint/llm"
        exit 2
        ;;
esac

input_file="${1:-$default_input}"
gpu_ids="${2:-$default_gpus}"
task="${TASK:-agent}"
eval_flag=(--eval)
if [[ "${EVAL:-1}" == "0" ]]; then
    eval_flag=()
fi

python infer_nlu.py \
    --backend "$backend" \
    --ckpt "$ckpt" \
    --input_file "$input_file" \
    --output_dir "$output_dir" \
    --gpu_ids "$gpu_ids" \
    --batch_size 32 \
    --max_new_tokens 256 \
    --task "$task" \
    "${eval_flag[@]}"
