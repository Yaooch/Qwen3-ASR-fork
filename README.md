# Qwen3-ASR 联合 CTC/RNNT 流式实验版

这个仓库基于原始 Qwen3-ASR，新增了 CTC/RNNT 辅助识别、联合训练、流式训练、流式推理和拼音评估能力。

核心目标：

- 复用 Qwen3-ASR 的音频 Encoder 和 LLM Decoder。
- 外接 CTC/RNNT 辅助头，做低延迟粗识别。
- RNNT/CTC 粗识别用于上屏、拼音检索、热词召回。
- 最终识别仍可由 LLM 生成，保证文本质量。
- 支持 chunk-wise 流式训练和流式推理实验。

## 新增功能

- `qwen_asr/joint/ctc.py`：CTC 辅助头。
- `qwen_asr/joint/rnnt.py`：RNNT 辅助头。
- `qwen_asr/joint/model.py`：联合模型，支持 `ctc/rnnt/llm/joint` 四种模式。
- `qwen_asr/joint/hotword.py`：基于粗识别文本的热词召回。
- `qwen_asr/joint/tokens.py`：辅助头词表和 SentencePiece 工具。
- `qwen_asr/tools/pinyin_eval.py`：拼音级 PER/SAR 评估。
- `finetuning/train.py`：联合训练入口。
- `finetuning/infer.py`：批量推理入口。
- `finetuning/train.sh`：训练启动脚本。
- `finetuning/eval.sh`：推理 + WER 评测脚本。

## 目录结构

```text
qwen_asr/
  joint/
    model.py      # 联合模型
    train_utils.py # 流式训练窗口
    stream.py     # 流式窗口和 chunk 特征
    decode.py     # CTC/RNNT/LLM/joint 推理
    ctc.py        # CTC 辅助头
    rnnt.py       # RNNT 辅助头
    hotword.py    # 热词召回
    tokens.py     # 词表工具
  tools/
    pinyin_eval.py

finetuning/
  train.py        # 训练入口
  infer.py        # 推理入口
  train.sh        # 训练脚本
  eval.sh         # 推理 + WER
  compute_pinyin.sh
  qwen3_asr_sft.py
```

## 辅助头接入位置

当前新训练固定使用：

```text
audio encoder 最后一层 Transformer -> ln_post -> CTC/RNNT
                                       -> proj1/proj2 -> LLM
```

旧 checkpoint 中如果带有历史字段，加载时仍会读取；新训练脚本不再暴露这些历史开关。

## 训练

```bash
bash finetuning/train.sh "0,1,2,3"
```

常用环境变量：

```bash
AUX_LOSS_TYPE=rnnt      # ctc / rnnt
AUX_ONLY=0             # 1 表示只训练辅助头
BATCH_SIZE=16
GRAD_ACC=4
QWEN_LR=2e-5
AUX_LR=1e-3
AUX_WEIGHT=0.3
```

流式训练：

```bash
AUX_STREAMING_TRAIN=1
AUX_STREAM_CHUNK_FRAMES=64
AUX_STREAM_LEFT_CONTEXT_FRAMES=64
AUX_STREAM_RIGHT_CONTEXT_FRAMES=7
AUX_STREAM_RANDOM_LEFT=1
bash finetuning/train.sh "0,1,2,3"
```

## 推理和 WER

```bash
bash finetuning/eval.sh \
  --ckpt /path/to/checkpoint \
  --mode rnnt \
  --input_scp /path/to/wav.scp \
  --ref_dir /path/to/text \
  --output_dir /path/to/out \
  --gpu_ids 0,1,2,3
```

流式推理：

```bash
bash finetuning/eval.sh --stream --mode rnnt
```

模式说明：

```text
ctc    只跑 CTC 辅助头
rnnt   只跑 RNNT 辅助头
llm    只跑 Qwen3-ASR LLM
joint  辅助头粗识别 + 热词召回 + LLM 最终识别
```

`--stream` 开启后：

- `ctc/rnnt`：按 chunk 输出粗识别。
- `llm`：按 chunk 计算音频 embedding，拼接后最终生成一次。
- `joint`：辅助头流式粗识别，LLM 使用 chunk embedding 做最终结果。

## 拼音评估

```bash
python -m qwen_asr.tools.pinyin_eval \
  --ref_path /path/to/ref \
  --result_path /path/to/results.txt \
  --output_path /path/to/pinyin.txt \
  --detail_output_path /path/to/pinyin_detail.jsonl \
  --badcase_path /path/to/pinyin_badcases.txt
```

主要指标：

- `sar`：拼音完全匹配的句子比例。
- `PER`：拼音 token 编辑错误率。
- `N/C/S/D/I`：参考 token 数、正确、替换、删除、插入。

## 当前原则

- `finetuning/` 只放训练、推理和评估入口。
- 核心联合模型代码放在 `qwen_asr/joint/`。
- 新增参数尽量少，历史实验开关不再继续暴露。
- RNNT 解码固定使用 cached greedy。
