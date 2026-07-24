# Finetuning 入口

`finetuning/` 只保留长期使用的训练、推理和评估入口；模型核心在 `qwen_asr/joint/`。

## 文件

```text
train.py             联合 ASR/CTC/RNNT 训练
infer.py             ASR 批量推理
train.sh             联合训练配置
infer.sh             推理 + WER + 拼音评估
infer_all.sh         多数据集批量推理
hotword_eval.sh      热词专项评估
grpo_train.py        热词 GRPO 训练
grpo_core.py         GRPO 数据、数学和文本 LoRA
grpo_train.sh        GRPO 启动配置
train_nlu.py         joint / 纯 Qwen3 文本 NLU SFT
infer_nlu.py         joint / 纯 Qwen3 文本 NLU 推理评测
train_nlu.sh         文本 NLU 启动配置
infer_nlu.sh         文本 NLU 推理配置
train_asr_nlu.py     ASR + ASR+NLU 联合 LoRA SFT
infer_asr_nlu.py     ASR+NLU 推理和 CER/意图评测
train_asr_nlu.sh     ASR+NLU 训练配置
infer_asr_nlu.sh     ASR+NLU 推理配置
```

## 联合 ASR 训练与推理

```bash
bash finetuning/train.sh "0,1,2,3"

bash finetuning/infer.sh \
  --ckpt /path/to/checkpoint \
  --mode llm,ctc \
  --input_scp /path/to/wav.scp \
  --ref_dir /path/to/text \
  --output_dir /path/to/out \
  --gpu_ids 0,1
```

`--mode` 支持 `llm/ctc/rnnt` 的逗号组合。`--encoder_mode` 支持
`offline/stream/train_mask`；`train_mask` 只是训练侧 chunk mask 理论上限，
线上真实流式评测使用 `stream`。

## 文本 NLU / Agent

文本任务统一使用 `train_nlu.py` 和 `infer_nlu.py`：

- `--backend joint`：加载 `Qwen3ASRJointModel`，无音频时直接走 thinker 文本 forward。
- `--backend llm`：加载纯 `AutoModelForCausalLM`。
- 两种 backend 共用数据、label mask、评测指标、badcase 和多 GPU 分片逻辑。

```bash
BACKEND=joint FULL_FT=0 bash finetuning/train_nlu.sh "0,1,2,3"
BACKEND=llm   FULL_FT=1 bash finetuning/train_nlu.sh "6,7"

BACKEND=joint TASK=agent EVAL=1 bash finetuning/infer_nlu.sh
BACKEND=llm   TASK=agent EVAL=1 bash finetuning/infer_nlu.sh
```

输入 JSONL 每行使用 `{"messages":[{system},{user},{assistant}]}`；推理也兼容
`{"text":"..."}`。大数据文件统一放在
`/cfs/data/private/WangYaoChi/train_data/all/nlu/`，不放仓库根目录。

## ASR + ASR+NLU

ASR+NLU 保留独立入口，因为它使用音频输入并把一条标注派生为两条训练样本：

- ASR：prompt=`转写语音`，target=`language X<asr_text>文本`
- ASR+NLU：prompt=`转写语音并提取用户意图`，target=`language X<asr_text>文本\n意图JSON`

```bash
bash finetuning/train_asr_nlu.sh "0,1,2,3"
bash finetuning/infer_asr_nlu.sh
```

训练只更新 thinker 文本 LoRA；推理输出文本 CER 和意图指标。

## 热词 GRPO

```bash
bash finetuning/grpo_train.sh [OUTPUT_DIR] [NPROC] [RESUME]
bash finetuning/hotword_eval.sh
```

GRPO 的 checkpoint、数据和输出路径只在 shell/CLI 中配置。`grpo_train.py`
不再保存另一套机器相关默认路径，并直接使用全部输入训练样本。
