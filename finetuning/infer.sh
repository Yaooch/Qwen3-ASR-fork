#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

# --------------------------------------------------
# 推理 + WER + 拼音评测脚本
#
# 功能：
# 1. 调用 infer.py 做推理
# 2. 输出 results_ctc.txt / results_rnnt.txt / results_llm.txt 和 details/results_detail.jsonl
# 3. 按 stage 调用 compute_asr_wer_with_slu.py 计算 WER
# 4. 按 stage 调用 qwen_asr.tools.pinyin_eval 计算拼音相似度
# --------------------------------------------------

# 1、/cfs/data/private/hubk/asr_test_set/VOYAH_Backflow(普通话回流测试集， 识别加语种)
# 2、/cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/{wav.scp,text} (回流粤语测试集，识别加语种)
# 3、/cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/{wav2.scp, text2} (回流四川测试集，识别加语种)
# 4、 /cfs/data/private/hubk/aishell_shard/chinese_test/{wav.scp,text}  (开源普通话测试集，识别加语种)
# 5、/cfs/data/private/hubk/asr_data/wenetspeech_yue/{2000.scp,2000.txt} （开源粤语测试集，识别加语种）
# 6、/cfs/data/private/hubk/asr_data/wenetspeech_sichuan/{10000.scp,10000.txt} (开源四川话测试集，识别加语种)
# 7、/cfs/data/private/hubk/asr_test_set/POI_ENTITY/{wav.scp,text}  (导航实体测试集，识别)
# 8、/cfs/data/private/hubk/asr_test_set/MEDIA_ENTITY/{wav.scp,text}  （媒体实体测试集，识别）
# 9、开放领域，测试集调研3个测试集(新闻、娱乐、美食、体育、军事、科技、汽车、法律、医疗、教育等)，摸一下开放领域效果(识别)

# 测试集:
# 川语:
# /cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/wav2.scp
# /cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/text2
# 普通话
# /cfs/data/private/hubk/asr_test_set/VOYAH_Backflow/nlu_wav.scp
# /cfs/data/private/hubk/asr_test_set/VOYAH_Backflow/nlu_text_classify

# aishell
# /cfs/data/private/hubk/aishell_shard/chinese_test/{wav.scp,text}

# aishell2
# /cfs/data/private/WangYaoChi/open_datasets/aishell2/AISHELL-DEV-TEST-SET/Mic/test/{wav.scp,trans.txt}

# Librispeech-clean
# /cfs/data/private/hubk/asr_test_set/Librispeech/librispeech/test/{wav.scp,text}

# eval
# /cfs/data/private/WangYaoChi/train_data/all/eval.scp
# /cfs/data/private/WangYaoChi/train_data/all/text
# 热词
# /cfs/data/private/hubk/asr_test_set/hotWords_test_set/wav.scp
# /cfs/data/private/hubk/asr_test_set/hotWords_test_set/text

CKPT="/cfs/data/private/WangYaoChi/model/qwen3-asr-ctc-joint-14/checkpoint-12653/"
# CKPT="/cfs/data/private/hubk/Qwen3-ASR/Qwen/Qwen3-ASR-1___7B"
STAGE="all"
MODE="ctc"
INPUT_SCP="/cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/wav2.scp"
REF_DIR="/cfs/data/private/hubk/asr_data/sichuan_yue_vehicle/text2"
OUTPUT_DIR="/cfs/data/private/WangYaoChi/test_out/joint_ctc_14/chuan/stream_64_132"
GPU_IDS="0,1,2,3,4,5,6,7"
BATCH_SIZE=128
DTYPE="bf16"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LANGUAGE=""
PROMPT=""
# PROMPT="你是一个拥有超高精度的语音识别引擎。专属名词列表如下:[孙作为, 宋雪倩, 薛思皓, 郭震, 户保坤, 岑吴镕, 王瑶池, 应臻奕, 淮水竹亭, 白月梵星, 清华池]。请根据音频内容进行识别，当遇到音素类似的词汇时，必须优先匹配列表中的专属名词，而不是通用词汇。"
HOTWORD_FILE=""
HOTWORD_TOPK=10
HOTWORD_PINYIN_STYLE="normal"
NO_AUX_IN_PROMPT=1
AUX_IN_PROMPT=0
RNNT_MAX_SYMBOLS_PER_STEP=3
AUX_ENCODER_BATCH_SIZE=5
STREAM=${STREAM:-1}
STREAM_CHUNK_SEC=${STREAM_CHUNK_SEC:-0.64}
STREAM_LEFT_CONTEXT_SEC=${STREAM_LEFT_CONTEXT_SEC:-1.32}
STREAM_RIGHT_CONTEXT_SEC=${STREAM_RIGHT_CONTEXT_SEC:-0.07}
STREAM_FIRST_CHUNK_LEFT_PAD_SEC=${STREAM_FIRST_CHUNK_LEFT_PAD_SEC:-0.0}
STREAM_WINDOW_BATCH_SIZE=${STREAM_WINDOW_BATCH_SIZE:-16}
STREAM_WINDOW_ENCODER_BATCH_SIZE=${STREAM_WINDOW_ENCODER_BATCH_SIZE:-4}

WER_SCRIPT="/root/scripts/compute_asr_wer_with_slu.py"
DETAILS_DIR="${OUTPUT_DIR}/details"
DOMAIN_PROMPT_FILE="${OUTPUT_DIR}/domain.txt"
WER_TXT_PATH="${DETAILS_DIR}/wer.txt"
PINYIN_OUTPUT_PATH="${OUTPUT_DIR}/pinyin_similarity.txt"
PINYIN_DETAIL_PATH="${DETAILS_DIR}/pinyin_detail.jsonl"
PINYIN_BADCASE_PATH="${DETAILS_DIR}/pinyin_badcases.txt"
PINYIN_STYLE="tone3"
PINYIN_KEEP_NON_CHINESE=0
PINYIN_TOPK_BADCASES=100
DOMAIN_PROMPT_FILE_SET=0
PINYIN_OUTPUT_PATH_SET=0
PINYIN_DETAIL_PATH_SET=0
PINYIN_BADCASE_PATH_SET=0

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
    echo "    --stage infer,wer \\"
    echo "    --domain_prompt_file /path/to/domain.txt"
    echo ""
    echo "参数说明："
    echo "  --stage               执行阶段：all / infer / wer / pinyin / eval，支持逗号组合；默认 infer,wer"
    echo "  --ckpt                Joint checkpoint 路径"
    echo "  --mode                推理模式：ctc / rnnt / llm / joint"
    echo "  --input_scp           输入 scp 文件"
    echo "  --output_dir          输出目录"
    echo "  --gpu_ids             使用的 GPU，例如 0 或 0,1,2,3"
    echo "  --batch_size          每个进程的 batch size"
    echo "  --dtype               模型精度：bf16 / fp16 / fp32"
    echo "  --python_bin          Python 可执行文件，默认 python3，也可用环境变量 PYTHON_BIN 覆盖"
    echo "  --language            默认语种，可不传"
    echo "  --prompt              额外提示词，仅 llm / joint 模式有效"
    echo "  --hotword_file        热词文件，可不传"
    echo "  --hotword_topk        热词召回数量"
    echo "  --hotword_pinyin_style 热词拼音召回风格：normal / tone3，默认 normal"
    echo "  --no_aux_in_prompt    joint 模式下不把 CTC/RNNT 结果注入 prompt，默认开启"
    echo "  --aux_in_prompt       joint 模式下把 CTC/RNNT 结果注入 prompt"
    echo "  --rnnt_max_symbols_per_step RNNT 每帧最多吐出的 token 数，调小可加速"
    echo "  --aux_encoder_batch_size CTC/RNNT audio encoder micro-batch，默认 1 最稳"
    echo "  --stream              使用 chunk-wise encoder 流式路径；llm/joint 会拼接 chunk audio embeddings"
    echo "  --no_stream           关闭流式推理；原始 Qwen3-ASR 模型需配合 --mode llm 使用"
    echo "  --stream_chunk_sec    流式当前 chunk 秒数"
    echo "  --stream_left_context_sec 流式左看秒数"
    echo "  --stream_right_context_sec 流式右看秒数"
    echo "  --stream_window_batch_size 流式窗口 batch size"
    echo "  --stream_window_encoder_batch_size 流式窗口 encoder micro-batch"
    echo "  --ref_dir             WER 参考文件路径"
    echo "  --domain_prompt_file  WER 脚本需要的 domain 文件"
    echo "  --wer_script          WER 脚本路径，默认 /root/scripts/compute_asr_wer_with_slu.py"
    echo "  --pinyin_output_path  拼音评估汇总输出，默认 output_dir/pinyin_similarity.txt"
    echo "  --pinyin_detail_path  拼音评估 jsonl 明细，默认 output_dir/pinyin_detail.jsonl"
    echo "  --pinyin_badcase_path 拼音评估 badcase，默认 output_dir/pinyin_badcases.txt"
    echo "  --pinyin_style        拼音风格：normal / tone3，默认 normal"
    echo "  --pinyin_keep_non_chinese 拼音评估保留英文数字 token"
    echo "  --pinyin_topk_badcases 拼音 badcase 输出条数，默认 100"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            STAGE="$2"
            shift 2
            ;;
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
        --python_bin)
            PYTHON_BIN="$2"
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
        --hotword_pinyin_style)
            HOTWORD_PINYIN_STYLE="$2"
            shift 2
            ;;
        --no_aux_in_prompt)
            NO_AUX_IN_PROMPT=1
            AUX_IN_PROMPT=0
            shift 1
            ;;
        --aux_in_prompt)
            AUX_IN_PROMPT=1
            NO_AUX_IN_PROMPT=0
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
        --no_stream)
            STREAM=0
            shift 1
            ;;
        --stream_chunk_sec)
            STREAM_CHUNK_SEC="$2"
            shift 2
            ;;
        --stream_left_context_sec)
            STREAM_LEFT_CONTEXT_SEC="$2"
            shift 2
            ;;
        --stream_right_context_sec)
            STREAM_RIGHT_CONTEXT_SEC="$2"
            shift 2
            ;;
        --stream_first_chunk_left_pad_sec)
            STREAM_FIRST_CHUNK_LEFT_PAD_SEC="$2"
            shift 2
            ;;
        --stream_window_batch_size)
            STREAM_WINDOW_BATCH_SIZE="$2"
            shift 2
            ;;
        --stream_window_encoder_batch_size)
            STREAM_WINDOW_ENCODER_BATCH_SIZE="$2"
            shift 2
            ;;
        --ref_dir)
            REF_DIR="$2"
            shift 2
            ;;
        --domain_prompt_file)
            DOMAIN_PROMPT_FILE="$2"
            DOMAIN_PROMPT_FILE_SET=1
            shift 2
            ;;
        --wer_script)
            WER_SCRIPT="$2"
            shift 2
            ;;
        --pinyin_output_path)
            PINYIN_OUTPUT_PATH="$2"
            PINYIN_OUTPUT_PATH_SET=1
            shift 2
            ;;
        --pinyin_detail_path)
            PINYIN_DETAIL_PATH="$2"
            PINYIN_DETAIL_PATH_SET=1
            shift 2
            ;;
        --pinyin_badcase_path)
            PINYIN_BADCASE_PATH="$2"
            PINYIN_BADCASE_PATH_SET=1
            shift 2
            ;;
        --pinyin_style)
            PINYIN_STYLE="$2"
            shift 2
            ;;
        --pinyin_keep_non_chinese)
            PINYIN_KEEP_NON_CHINESE=1
            shift 1
            ;;
        --pinyin_topk_badcases)
            PINYIN_TOPK_BADCASES="$2"
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

DETAILS_DIR="${OUTPUT_DIR}/details"
WER_TXT_PATH="${DETAILS_DIR}/wer.txt"
if [[ "${DOMAIN_PROMPT_FILE_SET}" -eq 0 ]]; then
    DOMAIN_PROMPT_FILE="${OUTPUT_DIR}/domain.txt"
fi
if [[ "${PINYIN_OUTPUT_PATH_SET}" -eq 0 ]]; then
    PINYIN_OUTPUT_PATH="${OUTPUT_DIR}/pinyin_similarity.txt"
fi
if [[ "${PINYIN_DETAIL_PATH_SET}" -eq 0 ]]; then
    PINYIN_DETAIL_PATH="${DETAILS_DIR}/pinyin_detail.jsonl"
fi
if [[ "${PINYIN_BADCASE_PATH_SET}" -eq 0 ]]; then
    PINYIN_BADCASE_PATH="${DETAILS_DIR}/pinyin_badcases.txt"
fi

RUN_INFER=0
RUN_WER=0
RUN_PINYIN=0

IFS=',' read -ra STAGE_ITEMS <<< "${STAGE}"
for item in "${STAGE_ITEMS[@]}"; do
    item="$(echo "${item}" | tr '[:upper:]' '[:lower:]' | xargs)"
    case "${item}" in
        all)
            RUN_INFER=1
            RUN_WER=1
            RUN_PINYIN=1
            ;;
        eval)
            RUN_WER=1
            RUN_PINYIN=1
            ;;
        infer)
            RUN_INFER=1
            ;;
        wer)
            RUN_WER=1
            ;;
        pinyin)
            RUN_PINYIN=1
            ;;
        "")
            ;;
        *)
            echo "错误：不支持的 stage: ${item}"
            echo "支持：all / infer / wer / pinyin / eval，或逗号组合如 infer,wer"
            exit 1
            ;;
    esac
done

if [[ "${RUN_INFER}" -eq 0 && "${RUN_WER}" -eq 0 && "${RUN_PINYIN}" -eq 0 ]]; then
    echo "错误：stage 为空，未选择任何执行阶段"
    exit 1
fi

if [[ "${RUN_INFER}" -eq 1 && -z "${CKPT}" ]]; then
    echo "错误：必须提供 --ckpt"
    exit 1
fi

if [[ "${RUN_INFER}" -eq 1 && -z "${INPUT_SCP}" ]]; then
    echo "错误：必须提供 --input_scp"
    exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "错误：必须提供 --output_dir"
    exit 1
fi

if [[ "${RUN_WER}" -eq 1 || "${RUN_PINYIN}" -eq 1 ]]; then
if [[ -z "${REF_DIR}" ]]; then
    echo "错误：必须提供 --ref_dir"
    exit 1
fi
fi

if [[ "${RUN_WER}" -eq 1 && -z "${DOMAIN_PROMPT_FILE}" ]]; then
    echo "错误：必须提供 --domain_prompt_file"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${DETAILS_DIR}"

INFER_CMD=(
    "${PYTHON_BIN}" infer.py
    --ckpt "${CKPT}"
    --mode "${MODE}"
    --input_scp "${INPUT_SCP}"
    --output_dir "${OUTPUT_DIR}"
    --gpu_ids "${GPU_IDS}"
    --batch_size "${BATCH_SIZE}"
    --dtype "${DTYPE}"
    --rnnt_max_symbols_per_step "${RNNT_MAX_SYMBOLS_PER_STEP}"
    --aux_encoder_batch_size "${AUX_ENCODER_BATCH_SIZE}"
    --stream_chunk_sec "${STREAM_CHUNK_SEC}"
    --stream_left_context_sec "${STREAM_LEFT_CONTEXT_SEC}"
    --stream_right_context_sec "${STREAM_RIGHT_CONTEXT_SEC}"
    --stream_first_chunk_left_pad_sec "${STREAM_FIRST_CHUNK_LEFT_PAD_SEC}"
    --stream_window_batch_size "${STREAM_WINDOW_BATCH_SIZE}"
    --stream_window_encoder_batch_size "${STREAM_WINDOW_ENCODER_BATCH_SIZE}"
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
    INFER_CMD+=(--hotword_pinyin_style "${HOTWORD_PINYIN_STYLE}")
fi

if [[ "${NO_AUX_IN_PROMPT}" -eq 1 ]]; then
    INFER_CMD+=(--no_aux_in_prompt)
fi
if [[ "${AUX_IN_PROMPT}" -eq 1 ]]; then
    INFER_CMD+=(--aux_in_prompt)
fi

if [[ "${STREAM}" -eq 1 ]]; then
    INFER_CMD+=(--stream)
fi

collect_result_targets() {
    RESULT_TARGETS=()
    if [[ -f "${OUTPUT_DIR}/results_ctc.txt" ]]; then
        RESULT_TARGETS+=("ctc:${OUTPUT_DIR}/results_ctc.txt")
    fi
    if [[ -f "${OUTPUT_DIR}/results_rnnt.txt" ]]; then
        RESULT_TARGETS+=("rnnt:${OUTPUT_DIR}/results_rnnt.txt")
    fi
    if [[ -f "${OUTPUT_DIR}/results_llm.txt" ]]; then
        RESULT_TARGETS+=("llm:${OUTPUT_DIR}/results_llm.txt")
    fi
}

if [[ "${RUN_INFER}" -eq 1 ]]; then
    echo "============================================================"
    echo "开始推理"
    echo "============================================================"
    echo "Stage: ${STAGE}"
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
    echo "Stream chunk/left/right: ${STREAM_CHUNK_SEC}/${STREAM_LEFT_CONTEXT_SEC}/${STREAM_RIGHT_CONTEXT_SEC}"
    echo "Stream first left pad: ${STREAM_FIRST_CHUNK_LEFT_PAD_SEC}"
    echo "Stream window batch: ${STREAM_WINDOW_BATCH_SIZE}"
    echo "Stream window encoder batch: ${STREAM_WINDOW_ENCODER_BATCH_SIZE}"
    echo "============================================================"

    "${INFER_CMD[@]}"
fi

collect_result_targets

if [[ "${RUN_WER}" -eq 1 || "${RUN_PINYIN}" -eq 1 ]]; then
if [[ "${#RESULT_TARGETS[@]}" -eq 0 ]]; then
    echo "错误：未找到推理结果文件 ${OUTPUT_DIR}/results_ctc.txt、results_rnnt.txt 或 results_llm.txt"
    echo "如果只跑评测，请先确认 output_dir 下已有 results_*.txt，或使用 --stage infer,wer"
    exit 1
fi
fi

if [[ "${RUN_WER}" -eq 1 ]]; then
echo "============================================================"
echo "开始计算 WER"
echo "============================================================"

for target in "${RESULT_TARGETS[@]}"; do
    suffix="${target%%:*}"
    result_path="${target#*:}"
    domain_path="${OUTPUT_DIR}/domain_${suffix}.txt"
    wer_path="${DETAILS_DIR}/wer_${suffix}.txt"
    echo "WER[${suffix}]: ${result_path}"
    "${PYTHON_BIN}" "${WER_SCRIPT}" \
        --char=1 \
        --v=1 \
        "${REF_DIR}" \
        "${result_path}" \
        "${domain_path}" \
        > "${wer_path}"
done
fi

if [[ "${RUN_PINYIN}" -eq 1 ]]; then
    echo "============================================================"
    echo "开始计算拼音评估"
    echo "============================================================"

    for target in "${RESULT_TARGETS[@]}"; do
        suffix="${target%%:*}"
        result_path="${target#*:}"
        if [[ "${suffix}" != "ctc" && "${suffix}" != "rnnt" ]]; then
            continue
        fi
        echo "拼音评估[${suffix}]: ${result_path}"
        PINYIN_CMD=(
            "${PYTHON_BIN}" "${PROJECT_ROOT}/qwen_asr/tools/pinyin_eval.py"
            --ref_path "${REF_DIR}"
            --result_path "${result_path}"
            --output_path "${OUTPUT_DIR}/pinyin_similarity_${suffix}.txt"
            --detail_output_path "${DETAILS_DIR}/pinyin_detail_${suffix}.jsonl"
            --badcase_path "${DETAILS_DIR}/pinyin_badcases_${suffix}.txt"
            --style "${PINYIN_STYLE}"
            --topk_badcases "${PINYIN_TOPK_BADCASES}"
        )

        if [[ "${PINYIN_KEEP_NON_CHINESE}" -eq 1 ]]; then
            PINYIN_CMD+=(--keep_non_chinese)
        fi

        "${PINYIN_CMD[@]}"
    done
fi

echo "============================================================"
echo "执行完成"
echo "============================================================"
echo "Stage: ${STAGE}"
echo "明细目录: ${DETAILS_DIR}"
echo "明细文件: ${DETAILS_DIR}/results_detail.jsonl"
echo "参考文件: ${REF_DIR}"
for target in "${RESULT_TARGETS[@]}"; do
    suffix="${target%%:*}"
    result_path="${target#*:}"
    echo "识别结果[${suffix}]: ${result_path}"
done
echo "WER文件: ${DETAILS_DIR}/wer_*.txt"
echo "Domain文件: ${OUTPUT_DIR}/domain_*.txt"
echo "拼音汇总: ${OUTPUT_DIR}/pinyin_similarity_*.txt"
echo "拼音明细: ${DETAILS_DIR}/pinyin_detail_*.jsonl"
echo "拼音Badcase: ${DETAILS_DIR}/pinyin_badcases_*.txt"
echo "============================================================"
