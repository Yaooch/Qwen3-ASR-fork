#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

stage="${1:-1}"
gpu_ids="${2:-0,1,4,5,6,7}"
num_gpus=$(awk -F',' '{print NF}' <<< "$gpu_ids")

base_model="/cfs/data/private/hubk/Qwen3-ASR/Qwen/Qwen3-ASR-1___7B"
train_file="/cfs/data/private/WangYaoChi/train_data/all/train_shuffled.jsonl"
eval_file="/cfs/data/private/WangYaoChi/train_data/all/eval_shuffled.jsonl"
output_root="/cfs/data/private/WangYaoChi/model/joint_ctc_llm_glclap_2"
text_model="/cfs/data/private/WangYaoChi/model/glclap/bert-base-multilingual-uncased"
word_df="/cfs/data/private/WangYaoChi/train_data/all/english_word_df.json"

if [[ "$stage" == "1" ]]; then
    model_path="$base_model"
    output_dir="${output_root}/stage1_heads"
    train_tasks="ctc,glclap"
    batch_size=16
    grad_acc=1
    epochs=0.1
    lr_encoder=0
    lr_llm=0
    lr_proj=0
    lr_ctc=2e-3
    lr_glclap_text=2e-5
    lr_glclap_proj=2e-3
    calibration_steps=0
elif [[ "$stage" == "2" ]]; then
    model_path="${output_root}/stage1_heads"
    if [[ ! -f "${model_path}/joint_config.json" ]]; then
        echo "请先完成stage 1：${model_path}/joint_config.json不存在"
        exit 1
    fi
    output_dir="${output_root}/stage2_joint"
    train_tasks="llm,proj,encoder,ctc,glclap"
    batch_size=16
    grad_acc=4
    epochs=1
    lr_encoder=1e-5
    lr_llm=1e-5
    lr_proj=1e-5
    lr_ctc=1e-3
    lr_glclap_text=1e-5
    lr_glclap_proj=1e-3
    calibration_steps=100
else
    echo "stage只支持1或2"
    exit 1
fi

tmp_dir="/tmp/qwen3_asr_joint/stage${stage}"
logging_dir="${PROJECT_ROOT}/reports/tensorboard/joint_ctc_llm_glclap_2/stage${stage}"
mkdir -p "$output_dir" "${output_root}/dataset_cache" "$tmp_dir" "$logging_dir"
export CUDA_VISIBLE_DEVICES="$gpu_ids"
export OMP_NUM_THREADS=4
export HF_DATASETS_CACHE="${output_root}/dataset_cache"
export TMPDIR="$tmp_dir"

echo "stage=${stage} gpu=${gpu_ids} nproc=${num_gpus}"
echo "model=${model_path}"
echo "output=${output_dir}"
echo "tasks=${train_tasks} per_gpu_batch=${batch_size} grad_acc=${grad_acc}"

master_port=$(shuf -n 1 -i 20000-65000)
torchrun \
    --nproc_per_node="$num_gpus" \
    --master_port="$master_port" \
    -m qwen_asr_ext.joint.train \
    --model_path "$model_path" \
    --train_file "$train_file" \
    --eval_file "$eval_file" \
    --output_dir "$output_dir" \
    --resume 0 \
    --train "$train_tasks" \
    --batch_size "$batch_size" \
    --grad_acc "$grad_acc" \
    --epochs "$epochs" \
    --stream_train 1 \
    --lr_encoder "$lr_encoder" \
    --lr_llm "$lr_llm" \
    --lr_proj "$lr_proj" \
    --lr_ctc "$lr_ctc" \
    --lr_glclap_text "$lr_glclap_text" \
    --lr_glclap_proj "$lr_glclap_proj" \
    --w_llm 1 \
    --w_ctc 1 \
    --w_glclap 1 \
    --loss_calibration_steps "$calibration_steps" \
    --ctc_grad_ratio 0.25 \
    --glclap_grad_ratio 0.25 \
    --glclap_text_model "$text_model" \
    --english_word_df "$word_df" \
    --max_subtext_units 8 \
    --save_steps 5000 \
    --log_steps 10 \
    --logging_dir "$logging_dir" \
    --num_workers 4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05
