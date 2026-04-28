#!/bin/bash
# CTC训练脚本 - 支持单卡/多卡训练

# 默认配置
# MODEL_PATH="/cfs/data/private/WangYaoChi/model/qwen3-asr-finetuning-out-3/checkpoint-9375"
MODEL_PATH="/cfs/data/private/hubk/Qwen3-ASR/Qwen/Qwen3-ASR-1___7B"
TRAIN_MANIFEST="/cfs/data/private/WangYaoChi/train_data/all/train_9.6w_shuffled.jsonl"
VAL_MANIFEST="/cfs/data/private/WangYaoChi/train_data/all/eval.jsonl"
VOCAB_PATH="ctc_vocab_2.json"
OUTPUT_DIR="/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc"

# 训练参数
BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=4
NUM_EPOCHS=20
LR=1e-3
NUM_WORKERS=4

# 音频过滤参数
MAX_AUDIO_DURATION=30.0
MIN_AUDIO_DURATION=0.5

# GPU设置
GPU_ID=0          # 默认使用第0张卡
USE_MULTI_GPU=0   # 默认单卡
NUM_GPUS=8        # 多卡时的卡数

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --train_manifest)
            TRAIN_MANIFEST="$2"
            shift 2
            ;;
        --val_manifest)
            VAL_MANIFEST="$2"
            shift 2
            ;;
        --vocab_path)
            VOCAB_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --gpu_id)
            GPU_ID="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            USE_MULTI_GPU=1
            shift 2
            ;;
        --max_audio_duration)
            MAX_AUDIO_DURATION="$2"
            shift 2
            ;;
        --min_audio_duration)
            MIN_AUDIO_DURATION="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "必需参数:"
            echo "  --train_manifest PATH       训练数据清单路径 (JSONL)"
            echo ""
            echo "可选参数:"
            echo "  --model_path PATH           模型路径 (默认: Qwen/Qwen3-ASR-1.7B)"
            echo "  --val_manifest PATH         验证数据清单路径"
            echo "  --vocab_path PATH           词表路径 (默认: ctc_vocab.json)"
            echo "  --output_dir PATH           输出目录 (默认: ./ctc_model_output)"
            echo "  --batch_size INT            每卡batch size (默认: 16)"
            echo "  --gradient_accumulation_steps INT 梯度累积步数 (默认: 4)"
            echo "  --num_epochs INT            训练轮数 (默认: 10)"
            echo "  --lr FLOAT                  学习率 (默认: 1e-4)"
            echo "  --gpu_id INT                单卡训练时使用的GPU ID (默认: 0)"
            echo "  --num_gpus INT              多卡训练时的卡数 (如: 8)"
            echo "  --max_audio_duration FLOAT  最大音频时长秒数 (默认: 30.0)"
            echo "  --min_audio_duration FLOAT  最小音频时长秒数 (默认: 0.5)"
            echo "  --num_workers INT           数据加载线程数 (默认: 4)"
            echo ""
            echo "示例:"
            echo "  # 单卡训练 (使用第0张卡)"
            echo "  $0 --train_manifest train.jsonl --val_manifest val.jsonl"
            echo ""
            echo "  # 单卡训练 (使用第1张卡)"
            echo "  $0 --train_manifest train.jsonl --gpu_id 1"
            echo ""
            echo "  # 8卡训练"
            echo "  $0 --train_manifest train.jsonl --num_gpus 8"
            echo ""
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$TRAIN_MANIFEST" ]; then
    echo "错误: 必须提供 --train_manifest 参数"
    echo "使用 -h 或 --help 查看帮助"
    exit 1
fi

# 构建命令参数
COMMON_ARGS="\
    --model_path $MODEL_PATH \
    --train_manifest $TRAIN_MANIFEST \
    --vocab_path $VOCAB_PATH \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRADIENT_ACCUMULATION_STEPS \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --output_dir $OUTPUT_DIR \
    --max_duration $MAX_AUDIO_DURATION \
    --num_workers $NUM_WORKERS"

# 添加可选参数
if [ -n "$VAL_MANIFEST" ]; then
    COMMON_ARGS="$COMMON_ARGS --val_manifest $VAL_MANIFEST"
fi



echo "============================================"
echo "CTC训练配置"
echo "============================================"
echo "模型路径: $MODEL_PATH"
echo "训练数据: $TRAIN_MANIFEST"
echo "验证数据: ${VAL_MANIFEST:-无}"
echo "词表路径: $VOCAB_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "梯度累积: $GRADIENT_ACCUMULATION_STEPS"
echo "训练轮数: $NUM_EPOCHS"
echo "学习率: $LR"

if [ $USE_MULTI_GPU -eq 1 ]; then
    echo "训练模式: 多卡训练 (${NUM_GPUS}张卡)"
    echo "============================================"
    

    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --master_port=29500 \
        train_ctc.py \
        $COMMON_ARGS
else
    echo "训练模式: 单卡训练 (GPU ${GPU_ID})"
    echo "============================================"
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    python train_ctc.py \
        $COMMON_ARGS
fi

echo "============================================"
echo "训练完成!"
echo "输出目录: $OUTPUT_DIR"
echo "============================================"