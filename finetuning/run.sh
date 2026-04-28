#!/usr/bin/env bash
# export HF_DATASETS_CACHE="/cfs/data/private/WangYaoChi/hf_cache/rank${LOCAL_RANK}"
# mkdir -p $HF_DATASETS_CACHE
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH="/cfs/data/private/hubk/Qwen3-ASR/Qwen/Qwen3-ASR-1___7B"
#TRAIN_FILE="/cfs/data/private/WangYaoChi/train_data/all/train_480w_shuffled.jsonl"
TRAIN_FILE="/cfs/data/private/hubk/Qwen3-ASR/Qwen/train_data/train.jsonl"
#EVAL_FILE="/cfs/data/private/WangYaoChi/train_data/all/eval.jsonl"
EVAL_FILE="/cfs/data/private/hubk/Qwen3-ASR/Qwen/train_data/eval.jsonl"
OUTPUT_DIR="/cfs/data/private/hubk/Qwen3-ASR/Qwen/qwen3-asr-finetuning2-out/"

torchrun --nproc_per_node=4 qwen3_asr_sft.py \
  --model_path ${MODEL_PATH} \
  --train_file ${TRAIN_FILE} \
  --eval_file ${EVAL_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --resume 1 \
  --batch_size 16 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --log_steps 50 \
  --save_strategy steps \
  --save_steps 300 \
  --save_total_limit 5 \
  --num_workers 64 \
  --pin_memory 1 \
  --persistent_workers 1 \
  --prefetch_factor 4 \

