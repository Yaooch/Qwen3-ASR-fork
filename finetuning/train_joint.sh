#!/bin/bash

# ==============================================================================
# Qwen3-ASR + CTC/RNNT 联合微调启动脚本
# 用法:
#   CTC:  bash train_joint.sh "0,1"
#   RNNT: AUX_LOSS_TYPE=rnnt bash train_joint.sh "0,1"
# ==============================================================================

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
TRAIN_FILE="/cfs/data/private/WangYaoChi/train_data/all/train_700w_shuffled.jsonl"
EVAL_FILE="/cfs/data/private/WangYaoChi/train_data/all/eval_shuffled.jsonl"
OUTPUT_DIR="/cfs/data/private/WangYaoChi/model/qwen3-asr-rnnt-2"

VOCAB_PATH="/nfsdir/hubk/sensevoice_training/wenet/examples/voyah/s0/data/dict/lang_char_large_yue.txt"
SP_MODEL_PATH="/nfsdir/hubk/sensevoice_training/wenet/examples/voyah/s0/data/dict/train_960_unigram5000.model"

# ==================== 训练参数 ====================
# 实际总 batch = BATCH_SIZE * GRAD_ACC * NUM_GPUS
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACC=${GRAD_ACC:-4}
EPOCHS=${EPOCHS:-1}

# aux_loss_type 可选 ctc / rnnt。RNNT 显存更高，建议先把 BATCH_SIZE 调小。
AUX_LOSS_TYPE=${AUX_LOSS_TYPE:-rnnt}
# Aux-only warmup: freeze Qwen, skip LLM forward, train only the auxiliary head.
AUX_ONLY=${AUX_ONLY:-0}
QWEN_LR=${QWEN_LR:-2e-5}
AUX_LR=${AUX_LR:-1e-3}
AUX_WEIGHT=${AUX_WEIGHT:-0.3}
AUX_ENCODER_BATCH_SIZE=${AUX_ENCODER_BATCH_SIZE:-4}

# post_proj 不需要关心 ctc_layer_idx；pre_proj 时才会使用该层号。
CTC_POSITION=${CTC_POSITION:-pre_proj}
CTC_LAYER_IDX=${CTC_LAYER_IDX:-24}

SAVE_STEPS=${SAVE_STEPS:-1000}
NUM_WORKERS=${NUM_WORKERS:-4}
MASTER_PORT=$(shuf -n 1 -i 20000-65000)

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    train_joint.py \
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
    --aux_encoder_batch_size "$AUX_ENCODER_BATCH_SIZE" \
    --ctc_position "$CTC_POSITION" \
    --ctc_layer_idx "$CTC_LAYER_IDX" \
    --save_steps "$SAVE_STEPS" \
    --log_steps 10 \
    --num_workers "$NUM_WORKERS" \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05
