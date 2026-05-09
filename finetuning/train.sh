#!/bin/bash

# ==============================================================================
# Qwen3-ASR + CTC/RNNT 联合微调启动脚本
# 用法:
#   CTC:  bash train.sh "0,1"
#   RNNT: AUX_LOSS_TYPE=rnnt bash train.sh "0,1"
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

GPU_IDS=${1:-"0,1,2,3,4,5,6,7"}
NUM_GPUS=$(echo "$GPU_IDS" | awk -F',' '{print NF}')

echo "==========================================================="
echo "  启动 Qwen3-ASR 联合微调"
echo "  指定使用显卡 ID : $GPU_IDS"
echo "  检测到 GPU 数量 : $NUM_GPUS"
echo "==========================================================="

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export OMP_NUM_THREADS=4

# ==================== 路径与数据 ====================
MODEL_PATH="/cfs/data/private/hubk/Qwen3-ASR/Qwen/Qwen3-ASR-1___7B"
# MODEL_PATH="/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14/checkpoint-12653/"
TRAIN_FILE="/cfs/data/private/WangYaoChi/train_data/all/train_700w_shuffled.jsonl"
EVAL_FILE="/cfs/data/private/WangYaoChi/train_data/all/eval_shuffled.jsonl"
OUTPUT_DIR="/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-15"

VOCAB_PATH="/nfsdir/hubk/sensevoice_training/wenet/examples/voyah/s0/data/dict/lang_char_large_yue.txt"
SP_MODEL_PATH="/nfsdir/hubk/sensevoice_training/wenet/examples/voyah/s0/data/dict/train_960_unigram5000.model"

# ==================== 训练参数 ====================
# 实际总 batch = BATCH_SIZE * GRAD_ACC * NUM_GPUS
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACC=${GRAD_ACC:-4}
EPOCHS=${EPOCHS:-2}

# 辅助损失类型：ctc / rnnt。RNNT 更占显存。
AUX_LOSS_TYPE=${AUX_LOSS_TYPE:-ctc}
# 只使用辅助损失，跳过 LLM forward。
AUX_ONLY=${AUX_ONLY:-1}
# AUX_ONLY=1 时，是否同时训练 audio encoder。
AUX_TRAIN_ENCODER=${AUX_TRAIN_ENCODER:-1}

QWEN_LR=${QWEN_LR:-2e-5}
AUX_LR=${AUX_LR:-2e-3}
AUX_WEIGHT=${AUX_WEIGHT:-0.1}
AUX_ENCODER_BATCH_SIZE=${AUX_ENCODER_BATCH_SIZE:-4}
AUX_STREAMING_TRAIN=${AUX_STREAMING_TRAIN:-0}
AUX_STREAM_CHUNK_FRAMES=${AUX_STREAM_CHUNK_FRAMES:-64}
AUX_STREAM_LEFT_CONTEXT_FRAMES=${AUX_STREAM_LEFT_CONTEXT_FRAMES:-128}
AUX_STREAM_RIGHT_CONTEXT_FRAMES=${AUX_STREAM_RIGHT_CONTEXT_FRAMES:-7}
AUX_STREAM_RANDOM_LEFT=${AUX_STREAM_RANDOM_LEFT:-1}
AUX_STREAM_WINDOW_BATCH_SIZE=${AUX_STREAM_WINDOW_BATCH_SIZE:-32}
AUDIO_N_WINDOW=${AUDIO_N_WINDOW:-0}
AUDIO_N_WINDOW_INFER=${AUDIO_N_WINDOW_INFER:-200}

SAVE_STEPS=${SAVE_STEPS:-1000}
NUM_WORKERS=${NUM_WORKERS:-4}
MASTER_PORT=$(shuf -n 1 -i 20000-65000)

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    train.py \
    --model_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --resume 1 \
    --output_dir "$OUTPUT_DIR" \
    --vocab_path "$VOCAB_PATH" \
    --sp_model_path "$SP_MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --grad_acc "$GRAD_ACC" \
    --qwen_lr "$QWEN_LR" \
    --ctc_lr "$AUX_LR" \
    --epochs "$EPOCHS" \
    --ctc_weight "$AUX_WEIGHT" \
    --aux_loss_type "$AUX_LOSS_TYPE" \
    --aux_only "$AUX_ONLY" \
    --aux_train_encoder "$AUX_TRAIN_ENCODER" \
    --aux_encoder_batch_size "$AUX_ENCODER_BATCH_SIZE" \
    --aux_streaming_train "$AUX_STREAMING_TRAIN" \
    --aux_stream_chunk_frames "$AUX_STREAM_CHUNK_FRAMES" \
    --aux_stream_left_context_frames "$AUX_STREAM_LEFT_CONTEXT_FRAMES" \
    --aux_stream_right_context_frames "$AUX_STREAM_RIGHT_CONTEXT_FRAMES" \
    --aux_stream_random_left "$AUX_STREAM_RANDOM_LEFT" \
    --aux_stream_window_batch_size "$AUX_STREAM_WINDOW_BATCH_SIZE" \
    --audio_n_window "$AUDIO_N_WINDOW" \
    --audio_n_window_infer "$AUDIO_N_WINDOW_INFER" \
    --save_steps "$SAVE_STEPS" \
    --log_steps 10 \
    --num_workers "$NUM_WORKERS" \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05
