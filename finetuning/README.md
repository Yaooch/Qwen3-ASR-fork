# Finetuning 入口

这个目录只保留训练、推理和评估脚本。核心模型代码在 `qwen_asr/joint/`。

## 文件

```text
train.py          联合训练入口
infer.py          批量推理入口
train.sh          训练启动脚本
infer.sh          推理 + WER + 拼音评估脚本
infer_all.sh      多数据集批量推理和汇总脚本
hotword_eval.sh   热词专项评估脚本
qwen3_asr_sft.py  原始 SFT baseline
train_nlu.py      NLU(用户意图提取)LoRA SFT 入口
infer_nlu.py      NLU 文本推理 + 意图评测入口
train_nlu.sh      NLU 训练启动脚本
infer_nlu.sh      NLU 推理/评测启动脚本
```

## 训练

```bash
bash train.sh "0,1,2,3"
```

常用配置直接改 `train.sh` 顶部变量。`--train` 只控制训练和 loss，不训练的已有 CTC/RNNT 头不加载，保存时原样复制。Audio window 使用模型配置默认值。
默认词表路径、SentencePiece 路径和 WER 脚本路径在 `qwen_asr/joint/defaults.py` 中维护。

## 推理

```bash
bash infer.sh \
  --ckpt /path/to/checkpoint \
  --mode llm,ctc \
  --input_scp /path/to/wav.scp \
  --ref_dir /path/to/text \
  --output_dir /path/to/out \
  --gpu_ids 0,1
```

`--mode` 只支持 `llm/ctc/rnnt` 的逗号组合，不再支持 `joint`。`--encoder_mode` 支持 `offline/stream/train_mask`：`offline` 使用整条音频离线前向，`stream` 使用真实 chunk-wise 流式路径，`train_mask` 使用与流式训练一致的整条 Mel + chunk mask Encoder 路径评测理论上限。`--stream/--no_stream` 分别兼容映射到 `stream/offline`。具体窗口参数在 `qwen_asr/joint/defaults.py` 中修改。

热词评估会同时输出默认 LLM 和热词 prompt LLM，并比较两者的热词识别变化。

## 批量推理

```bash
bash infer_all.sh --ckpt /path/to/checkpoint --datasets_file datasets.txt --outdir /path/to/out --encoder_mode train_mask
```

默认使用脚本内置数据集列表；传 `--datasets_file` 可覆盖。`datasets.txt` 每行格式：`name|wav.scp|text|language`。`infer_all.sh` 默认使用 `train_mask`，用于先观察与流式训练严格对齐时的评测结果；线上真实流式评测显式传 `--encoder_mode stream`。

## NLU（用户意图提取）

NLU 是纯文本任务（user 语句 → 意图 JSON），与 ASR 共存于同一 LLM：基线 joint checkpoint 冻结，LoRA 打在 thinker 文本解码器（复用 `grpo_core.apply_lora`），NLU 用独立的标准 Qwen chat format prompt（不走 ASR 的 audio chat_template），system prompt（"提取用户意图"）做任务路由。`joint.forward` 检测到 `input_features` 为空时走纯文本 thinker 前向分支，ASR 路径不受影响。

```bash
bash train_nlu.sh "0,1,2,3"                       # 训练
bash infer_nlu.sh <input.jsonl> 0                 # 推理
EVAL=1 bash infer_nlu.sh <input_with_ref.jsonl> 0 # 评测（输出 name_acc / args_exact / json_valid）
```

输入 jsonl 每行 `{"messages":[{system},{user},{assistant}]}`（`assistant` 可选；`--eval` 时作 ref），也支持 `{"text":"..."}`。默认基线 / LoRA / 数据路径在 `train_nlu.sh`、`infer_nlu.sh` 顶部。意图 prompt 构造与评测指标工具在 `qwen_asr/tools/nlu.py`。本期只验证 NLU，为后期三能力（ASR / NLU / ASR+NLU）统一留接口。
