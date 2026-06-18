#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

python_bin="${PYTHON_BIN:-python3}"
gpu_ids="0,1,2,3,4,5,6,7"

model_path="/cfs/data/private/WangYaoChi/model/joint_ctc_50"
train_file="/cfs/data/private/WangYaoChi/train_data/all/contextasr/train_contextasr2.jsonl"
eval_file="/cfs/data/private/WangYaoChi/train_data/all/contextasr/eval_contextasr.jsonl"
output_dir="/cfs/data/private/WangYaoChi/model/joint_ctc_50_hotword_scorer_2"

sr=16000
batch_size=32
epochs=5
lr=1e-4
weight_decay=0.01
pos_weight=1.0
threshold=0.5
max_audio_sec=100000
max_hotword_len=24
scorer_dim=384
num_heads=8
num_layers=2
ffn_mult=2
dropout=0.1
chunk_hotwords=256
num_workers=0
log_steps=20
logging_dir="./logs/logs_ctc_50_hotword_scorer_2"

declare -A arg_map=(
    [--python_bin]=python_bin [--gpu_ids]=gpu_ids
    [--model_path]=model_path [--train_file]=train_file [--eval_file]=eval_file
    [--output_dir]=output_dir [--sr]=sr [--batch_size]=batch_size [--epochs]=epochs
    [--lr]=lr [--weight_decay]=weight_decay [--pos_weight]=pos_weight [--threshold]=threshold
    [--max_audio_sec]=max_audio_sec [--max_hotword_len]=max_hotword_len
    [--scorer_dim]=scorer_dim [--num_heads]=num_heads [--num_layers]=num_layers
    [--ffn_mult]=ffn_mult [--dropout]=dropout
    [--chunk_hotwords]=chunk_hotwords [--num_workers]=num_workers [--log_steps]=log_steps
    [--logging_dir]=logging_dir
)

usage() {
    echo "用法：bash $0 --model_path CKPT --train_file train.jsonl --output_dir out [options]"
    echo ""
    echo "必要参数："
    echo "  --model_path          joint checkpoint 路径"
    echo "  --train_file          训练 jsonl，需包含 audio/text/prompt"
    echo "  --output_dir          scorer 输出目录"
    echo ""
    echo "常用参数："
    echo "  --eval_file           验证 jsonl，可不传"
    echo "  --gpu_ids             使用 GPU，例如 0，默认 0"
    echo "  --python_bin          Python 解释器，默认 python3，也可用 PYTHON_BIN 覆盖"
    echo "  --batch_size          batch size，默认 32"
    echo "  --epochs              训练轮数，默认 1"
    echo "  --lr                  scorer 学习率，默认 1e-3"
    echo "  --pos_weight          BCE 正样本权重，默认 6.0"
    echo "  --threshold           训练日志和验证指标使用的概率阈值，默认 0.5"
    echo "  --max_hotword_len     热词 token 截断长度，默认 24"
    echo "  --chunk_hotwords      热词分块大小，默认 256"
    echo "  --max_audio_sec       超过该时长的音频跳过，默认 30.0；<=0 表示不限制"
    echo "  --logging_dir         TensorBoard 日志目录，默认 ${logging_dir}"
    echo ""
}

set_arg() {
    local opt="$1"
    local value="${2:-}"
    local var="${arg_map[$opt]:-}"

    if [[ -z "${value}" || "${value}" == --* ]]; then
        echo "参数 ${opt} 缺少取值"
        usage
        exit 1
    fi
    if [[ -z "${var}" ]]; then
        echo "未知参数: ${opt}"
        usage
        exit 1
    fi
    printf -v "${var}" '%s' "${value}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            set_arg "$1" "${2:-}"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${model_path}" ]]; then
    echo "错误：必须提供 --model_path"
    exit 1
fi
if [[ -z "${train_file}" ]]; then
    echo "错误：必须提供 --train_file"
    exit 1
fi
if [[ -z "${output_dir}" ]]; then
    echo "错误：必须提供 --output_dir"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "${output_dir}"
num_gpus=$(echo "${gpu_ids}" | awk -F',' '{print NF}')
master_port=$(shuf -n 1 -i 20000-65000)

train_args=(
    --model_path "${model_path}"
    --train_file "${train_file}"
    --output_dir "${output_dir}"
    --sr "${sr}"
    --batch_size "${batch_size}"
    --epochs "${epochs}"
    --lr "${lr}"
    --weight_decay "${weight_decay}"
    --pos_weight "${pos_weight}"
    --threshold "${threshold}"
    --max_audio_sec "${max_audio_sec}"
    --max_hotword_len "${max_hotword_len}"
    --scorer_dim "${scorer_dim}"
    --num_heads "${num_heads}"
    --num_layers "${num_layers}"
    --ffn_mult "${ffn_mult}"
    --dropout "${dropout}"
    --chunk_hotwords "${chunk_hotwords}"
    --num_workers "${num_workers}"
    --log_steps "${log_steps}"
    --logging_dir "${logging_dir}"
)

if [[ -n "${eval_file}" ]]; then
    train_args+=(--eval_file "${eval_file}")
fi

if [[ "${num_gpus}" -gt 1 ]]; then
    cmd=(
        "${python_bin}" -m torch.distributed.run
        --nproc_per_node "${num_gpus}"
        --master_port "${master_port}"
        train_hotword_scorer.py
        "${train_args[@]}"
    )
else
    cmd=("${python_bin}" train_hotword_scorer.py "${train_args[@]}")
fi

echo "============================================================"
echo "训练 Encoder 热词打分器"
echo "模型：${model_path}"
echo "训练：${train_file}"
echo "验证：${eval_file:-无}"
echo "输出：${output_dir}"
echo "GPU：${gpu_ids}"
echo "GPU 数：${num_gpus}"
echo "每卡 batch：${batch_size}"
echo "有效 batch：$((batch_size * num_gpus))"
echo "Python：${python_bin}"
echo "日志：${logging_dir}"
echo "============================================================"

"${cmd[@]}"
