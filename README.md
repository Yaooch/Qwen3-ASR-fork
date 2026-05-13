# Qwen3-ASR 联合 CTC/RNNT 流式实验版

这个仓库基于原始 Qwen3-ASR，新增了 CTC/RNNT 辅助识别、联合训练、流式推理和拼音评估能力。

核心目标：

- 复用 Qwen3-ASR 的音频 Encoder 和 LLM Decoder。
- 外接 CTC/RNNT 辅助头，做低延迟粗识别。
- RNNT/CTC 粗识别用于上屏、拼音检索、热词召回。
- 最终识别仍可由 LLM 生成，保证文本质量。
- 支持固定窗口的流式推理实验。

## 新增功能

- `qwen_asr/joint/ctc.py`：CTC 辅助头。
- `qwen_asr/joint/rnnt.py`：RNNT 辅助头。
- `qwen_asr/joint/model.py`：联合模型，支持 `llm/ctc/rnnt` 逗号组合。
- `qwen_asr/joint/hotword.py`：基于粗识别文本的热词召回。
- `qwen_asr/joint/tokens.py`：辅助头词表和 SentencePiece 工具。
- `qwen_asr/tools/pinyin_eval.py`：拼音级 PER/SAR 评估。
- `finetuning/train.py`：联合训练入口。
- `finetuning/infer.py`：批量推理入口。
- `finetuning/train.sh`：训练启动脚本。
- `finetuning/infer.sh`：推理 + WER + 拼音评估脚本。
- `finetuning/hotword_eval.sh`：热词 prompt 评估脚本。

## 目录结构

```text
qwen_asr/
  joint/
    model.py      # 联合模型
    defaults.py   # 默认提示词和内部推理常量
    stream.py     # 流式窗口和 chunk 特征
    decode.py     # CTC/RNNT/LLM 推理
    ctc.py        # CTC 辅助头
    rnnt.py       # RNNT 辅助头
    hotword.py    # 热词召回
    tokens.py     # 词表工具
  tools/
    hotword_eval.py
    pinyin_eval.py

finetuning/
  train.py        # 训练入口
  infer.py        # 推理入口
  train.sh        # 训练脚本
  infer.sh        # 推理 + WER + 拼音
  infer_all.sh    # 多数据集推理
  hotword_eval.sh # 热词评估
  qwen3_asr_sft.py
```

## 辅助头接入位置

当前新训练固定使用：

```text
audio encoder 最后一层 Transformer -> ln_post -> CTC/RNNT
                                       -> proj1/proj2 -> LLM
```

新 joint checkpoint 使用 `joint_config.json` 记录 CTC/RNNT 头；`--train` 只控制训练和 loss，不训练的已有头不加载，保存时从源 checkpoint 复制。

## 训练

```bash
bash finetuning/train.sh "0,1,2,3"
```

常用配置直接改 `finetuning/train.sh` 顶部变量。训练任务用逗号组合：

```bash
train_tasks="llm,ctc"          # 可选 llm,encoder,ctc,rnnt
lr_llm=2e-5
lr_encoder=1e-5
lr_ctc=1e-3
lr_rnnt=1e-3
w_llm=1.0
w_ctc=0.3
w_rnnt=0.3
```

训练不再暴露额外窗口参数；短窗口由 `audio_n_window/audio_n_window_infer` 控制。
默认词表路径和 SentencePiece 路径在 `qwen_asr/joint/defaults.py` 中维护。

## 推理和 WER

```bash
bash finetuning/infer.sh \
  --ckpt /path/to/checkpoint \
  --mode llm,ctc \
  --input_scp /path/to/wav.scp \
  --ref_dir /path/to/text \
  --output_dir /path/to/out \
  --gpu_ids 0,1,2,3
```

流式推理：

```bash
bash finetuning/infer.sh --stream --mode llm,ctc
```

模式说明：

```text
llm       只跑 Qwen3-ASR LLM
ctc       只跑 CTC 辅助头
rnnt      只跑 RNNT 辅助头
llm,ctc   同时输出默认 LLM 和 CTC；有热词文件时用 CTC 召回热词再跑热词 LLM
```

`--stream` 开启后：

- `ctc/rnnt`：按 chunk 输出粗识别。
- `llm`：按 chunk 计算音频 embedding，拼接后最终生成一次。

## 拼音评估

```bash
python -m qwen_asr.tools.pinyin_eval \
  --ref_path /path/to/ref \
  --result_path /path/to/results.txt \
  --output_path /path/to/pinyin.txt \
  --badcase_path /path/to/pinyin_badcases.txt
```

主要指标：

- `sar`：拼音完全匹配的句子比例。
- `PER`：拼音 token 编辑错误率。
- `SUB/INS/DEL/REF`：替换、插入、删除、参考 token 数。

## 当前原则

- `finetuning/` 只放训练、推理和评估入口。
- 核心联合模型代码放在 `qwen_asr/joint/`。
- 新增参数尽量少，历史实验开关不再继续暴露。
- RNNT 解码固定使用 cached greedy。
