#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

# --------------------------------------------------
# 推理 + WER 评测脚本
#
# 功能：
# 1. 调用 infer.py 做推理
# 2. 输出 results.txt 和 results_detail.jsonl
# 3. 自动调用 compute_asr_wer_with_slu.sh 计算 WER
# --------------------------------------------------

# 测试集:
# 川语:
# /cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/wav2.scp
# /cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/text2
# 普通话
# /cfs/data/private/hubk/asr_test_set/VOYAH_Backflow/nlu_wav.scp
# /cfs/data/private/hubk/asr_test_set/VOYAH_Backflow/nlu_text_classify
# eval
# /cfs/data/private/WangYaoChi/train_data/all/eval.scp
# /cfs/data/private/WangYaoChi/train_data/all/text
# 热词
# /cfs/data/private/hubk/asr_test_set/hotWords_test_set/wav.scp
# /cfs/data/private/hubk/asr_test_set/hotWords_test_set/text

CKPT="/cfs/data/private/WangYaoChi/model/qwen3-asr-rnnt-2/checkpoint-13283"
MODE="llm"
INPUT_SCP="/cfs/data/private/hubk/asr_test_set/VOYAH_Backflow/nlu_wav_2.scp"
REF_DIR="/cfs/data/private/hubk/asr_test_set/VOYAH_Backflow/nlu_text_classify_2"
OUTPUT_DIR="/cfs/data/private/WangYaoChi/test_out/joint2_llm_2/mandarin2"
GPU_IDS="6,7"
BATCH_SIZE=128
DTYPE="bf16"
LANGUAGE=""
PROMPT=""
# PROMPT="你是一个拥有超高精度的语音识别引擎。专属名词列表如下:[孙作为, 宋雪倩, 薛思皓, 应臻奕, 郭震, 户保坤, 岑吴镕, 王瑶池, 淮水竹亭, 白月梵星, 清华池]。请根据音频内容进行识别，当遇到音素类似的词汇时，必须优先匹配列表中的专属名词，而不是通用词汇。"
HOTWORD_FILE=""
HOTWORD_TOPK=10
NO_CTC_IN_PROMPT=0
RNNT_MAX_SYMBOLS_PER_STEP=3
AUX_ENCODER_BATCH_SIZE=5
STREAM=${STREAM:-1}

WER_SCRIPT="/root/scripts/compute_asr_wer_with_slu.py"
DOMAIN_PROMPT_FILE="${OUTPUT_DIR}/domain.txt"
WER_TXT_PATH="${OUTPUT_DIR}/wer.txt"

usage() {
    echo "用法："
    echo "  bash $0 \\"
    echo "    --ckpt /path/to/checkpoint \\"
    echo "    --mode joint \\"
    echo "    --input_scp /path/to/test.scp \\"
    echo "    --output_dir /path/to/output \\"
    echo "    --gpu_ids 0,1,2,3 \\"
    echo "    --batch_size 16 \\"
    echo "    --ref_dir /path/to/ref_dir \\"
    echo "    --domain_prompt_file /path/to/domain.txt"
    echo ""
    echo "参数说明："
    echo "  --ckpt                Joint checkpoint 路径"
    echo "  --mode                推理模式：ctc / rnnt / llm / joint"
    echo "  --input_scp           输入 scp 文件"
    echo "  --output_dir          输出目录"
    echo "  --gpu_ids             使用的 GPU，例如 0 或 0,1,2,3"
    echo "  --batch_size          每个进程的 batch size"
    echo "  --dtype               模型精度：bf16 / fp16 / fp32"
    echo "  --language            默认语种，可不传"
    echo "  --prompt              额外提示词，仅 llm / joint 模式有效"
    echo "  --hotword_file        热词文件，可不传"
    echo "  --hotword_topk        热词召回数量"
    echo "  --no_ctc_in_prompt    joint 模式下不把 CTC/RNNT 结果注入 prompt"
    echo "  --rnnt_max_symbols_per_step RNNT 每帧最多吐出的 token 数，调小可加速"
    echo "  --aux_encoder_batch_size CTC/RNNT audio encoder micro-batch，默认 1 最稳"
    echo "  --stream              使用 chunk-wise encoder 流式路径；llm/joint 会拼接 chunk audio embeddings"
    echo "  --ref_dir             WER 参考文件路径"
    echo "  --domain_prompt_file  WER 脚本需要的 domain 文件"
    echo "  --wer_script          WER 脚本路径，默认 /root/scripts/compute_asr_wer_with_slu.sh"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --input_scp)
            INPUT_SCP="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --hotword_file)
            HOTWORD_FILE="$2"
            shift 2
            ;;
        --hotword_topk)
            HOTWORD_TOPK="$2"
            shift 2
            ;;
        --no_ctc_in_prompt)
            NO_CTC_IN_PROMPT=1
            shift 1
            ;;
        --rnnt_max_symbols_per_step)
            RNNT_MAX_SYMBOLS_PER_STEP="$2"
            shift 2
            ;;
        --aux_encoder_batch_size)
            AUX_ENCODER_BATCH_SIZE="$2"
            shift 2
            ;;
        --stream)
            STREAM=1
            shift 1
            ;;
        --ref_dir)
            REF_DIR="$2"
            shift 2
            ;;
        --domain_prompt_file)
            DOMAIN_PROMPT_FILE="$2"
            shift 2
            ;;
        --wer_script)
            WER_SCRIPT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${CKPT}" ]]; then
    echo "错误：必须提供 --ckpt"
    exit 1
fi

if [[ -z "${INPUT_SCP}" ]]; then
    echo "错误：必须提供 --input_scp"
    exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "错误：必须提供 --output_dir"
    exit 1
fi

if [[ -z "${REF_DIR}" ]]; then
    echo "错误：必须提供 --ref_dir"
    exit 1
fi

if [[ -z "${DOMAIN_PROMPT_FILE}" ]]; then
    echo "错误：必须提供 --domain_prompt_file"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

INFER_CMD=(
    python infer.py
    --ckpt "${CKPT}"
    --mode "${MODE}"
    --input_scp "${INPUT_SCP}"
    --output_dir "${OUTPUT_DIR}"
    --gpu_ids "${GPU_IDS}"
    --batch_size "${BATCH_SIZE}"
    --dtype "${DTYPE}"
    --rnnt_max_symbols_per_step "${RNNT_MAX_SYMBOLS_PER_STEP}"
    --aux_encoder_batch_size "${AUX_ENCODER_BATCH_SIZE}"
)

if [[ -n "${LANGUAGE}" ]]; then
    INFER_CMD+=(--language "${LANGUAGE}")
fi

if [[ -n "${PROMPT}" ]]; then
    INFER_CMD+=(--prompt "${PROMPT}")
fi

if [[ -n "${HOTWORD_FILE}" ]]; then
    INFER_CMD+=(--hotword_file "${HOTWORD_FILE}")
    INFER_CMD+=(--hotword_topk "${HOTWORD_TOPK}")
fi

if [[ "${NO_CTC_IN_PROMPT}" -eq 1 ]]; then
    INFER_CMD+=(--no_ctc_in_prompt)
fi

if [[ "${STREAM}" -eq 1 ]]; then
    INFER_CMD+=(--stream)
fi

echo "============================================================"
echo "开始推理"
echo "============================================================"
echo "模型路径: ${CKPT}"
echo "推理模式: ${MODE}"
echo "输入文件: ${INPUT_SCP}"
echo "输出目录: ${OUTPUT_DIR}"
echo "GPU: ${GPU_IDS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "精度: ${DTYPE}"
echo "RNNT max_symbols_per_step: ${RNNT_MAX_SYMBOLS_PER_STEP}"
echo "Aux encoder batch size: ${AUX_ENCODER_BATCH_SIZE}"
echo "Stream: ${STREAM}"
echo "============================================================"

"${INFER_CMD[@]}"

RESULT_PATH="${OUTPUT_DIR}/results.txt"

if [[ ! -f "${RESULT_PATH}" ]]; then
    echo "错误：未找到推理结果文件 ${RESULT_PATH}"
    exit 1
fi

echo "============================================================"
echo "开始计算 WER"
echo "============================================================"

python "${WER_SCRIPT}" \
    --char=1 \
    --v=1 \
    "${REF_DIR}" \
    "${RESULT_PATH}" \
    "${DOMAIN_PROMPT_FILE}" \
    > "${WER_TXT_PATH}"


echo "============================================================"
echo "执行完成"
echo "============================================================"
echo "明细文件: ${OUTPUT_DIR}/results_detail.jsonl"
echo "参考文件: ${REF_DIR}"
echo "WER明细文件: ${WER_TXT_PATH}"
echo "识别结果: ${RESULT_PATH}"
echo "Domain文件: ${DOMAIN_PROMPT_FILE}"
echo "============================================================"
