# Qwen3-ASR 扩展代码

本目录集中存放项目自研能力。官方 `qwen_asr/` 保持上游结构；扩展代码只允许依赖官方包或同级扩展模块，官方包不反向依赖这里。

## 目录与职责

```text
qwen_asr_ext/
├── joint/                  # 联合 CTC/RNNT 模型、训练和推理主链路
│   ├── model.py            # Qwen3ASRJointModel、LLM 输入构造、transcribe
│   ├── encoder.py          # encoder 长度换算、离线/流式/train-mask 编码
│   ├── ctc.py              # CTC head、adapter、解码与 loss
│   ├── rnnt.py             # RNNT head、cached greedy 解码与 loss
│   ├── defaults.py         # prompt、路径、流式常量等共享默认值
│   ├── lora.py             # joint 文本解码器 LoRA 装配与校验
│   ├── train.py            # 联合训练入口
│   ├── infer.py            # 批量推理入口
│   └── scripts/            # train/infer/infer_all shell 入口
├── hotword/                # 音素热词检索与热词提示评测
│   ├── phoneme.py          # 文本音素化和边界约束 DP 精筛
│   ├── retriever.py        # FastRAG 粗筛和 HotwordRetriever 统一入口
│   ├── english.py          # 英文发音匹配
│   ├── evaluate.py         # 热词注入效果与延迟评测
│   └── evaluate.sh         # 热词推理加评测入口
├── grpo/                   # 热词 GRPO 强化学习
│   ├── grpo.py             # 样本读取、奖励、组内优势、PPO clip 与 KL
│   ├── train.py            # rollout、logprob 和分布式训练主循环
│   └── train.sh            # 训练入口
├── glclap/                 # 音频-文本热词检索
│   ├── README.md           # V1～V4 实验记录和后续方案
│   ├── model.py            # GLCLAP 模型和对比损失
│   ├── train.py            # 训练数据、采样和训练循环
│   ├── train.sh            # 训练入口
│   ├── benchmark.py        # 固定候选词表 benchmark 的数据协议与指标
│   ├── evaluate.py         # GLCLAP 音频查询评测
│   └── phoneme_baseline.py # 同一协议下的文本音素检索基线
├── nlu/                    # 文本 NLU 与 ASR+NLU
│   ├── common.py           # prompt、标签解析和指标
│   ├── train.py/infer.py   # 文本 NLU 的 joint/llm 双 backend
│   ├── train_asr.py        # ASR+NLU 音频联合训练
│   ├── infer_asr.py        # ASR+NLU 音频推理评测
│   └── scripts/            # 四个长期使用的 shell 入口
└── evaluation/             # 可复用的通用 ASR 评测
    ├── edit.py             # 两种评测共用的文本读取与编辑距离
    ├── pinyin.py           # 拼音 PER/SAR 与 badcase
    └── text_badcase.py     # 文本 WER badcase
```

核心调用关系是：`joint` 提供模型与编码主链路；`hotword` 只负责检索；`grpo` 训练 joint 的文本解码器 LoRA；`glclap` 是独立检索实验；`nlu` 复用 joint 模型和 LoRA 装配。LoRA 放在 `joint/lora.py`，因为 GRPO、文本 NLU 和 ASR+NLU 都会使用它。

## 从功能入口阅读代码

先看下面的入口总表，再沿对应调用链向下读。通常不需要先通读工具函数。

| 功能 | 推荐入口 | Python 主入口 | 最终核心 |
|---|---|---|---|
| Joint 训练 | `joint/scripts/train.sh` | `joint.train.main` | `Qwen3ASRJointModel.forward` |
| Joint 推理 | `joint/scripts/infer.sh` | `joint.infer.main` | `Qwen3ASRJointModel.transcribe` |
| 多数据集推理 | `joint/scripts/infer_all.sh` | 调用 `infer.sh` | 同上 |
| 热词推理与评测 | `hotword/evaluate.sh` | `joint.infer.main` + `hotword.evaluate.main` | `HotwordRetriever.retrieve` |
| GRPO 训练 | `grpo/train.sh` | `grpo.train.main` | `RolloutSampler` + `grpo_loss` |
| GLCLAP 训练 | `glclap/train.sh` | `glclap.train.main` | `GLCLAPModel.forward` |
| GLCLAP 评测 | Python 模块启动 | `glclap.evaluate.main` | `retrieve_batch` |
| 音素检索 baseline | Python 模块启动 | `glclap.phoneme_baseline.main` | `HotwordRetriever.retrieve` |
| 文本 NLU 训练/推理 | `nlu/scripts/train.sh` / `infer.sh` | `nlu.train.main` / `nlu.infer.main` | `NluCollator` / `worker` |
| ASR+NLU 训练/推理 | `nlu/scripts/train_asr.sh` / `infer_asr.sh` | `nlu.train_asr.main` / `nlu.infer_asr.main` | Joint `forward` / `transcribe` |
| 拼音与文本 badcase | 由 `joint/scripts/infer.sh` 调用 | `evaluation.*.main` | token 化、编辑距离与报告输出 |

### 1. Joint 训练

```text
joint/scripts/train.sh
→ joint.train.main
→ DataCollatorForJointTraining：音频特征、LLM labels、CTC/RNNT target
→ Qwen3ASRJointModel.forward
→ encoder.encode_offline 或 encoder.encode_train_mask
→ CTC.forward / RNNT.forward / thinker.forward
→ JointTrainer.compute_loss
→ JointTrainer.save_model
```

建议依次读 [joint/train.py](joint/train.py) 的 `main`、`DataCollatorForJointTraining`，然后读 [joint/model.py](joint/model.py) 的 `forward`。`forward` 是多任务汇合点：音频只编码一次，CTC、RNNT 和 LLM 再分别计算 loss。Encoder 的具体差异在 [joint/encoder.py](joint/encoder.py)，head 内部数学在 [joint/ctc.py](joint/ctc.py) 和 [joint/rnnt.py](joint/rnnt.py)。

### 2. Joint 批量推理

```text
joint/scripts/infer.sh
→ joint.infer.main：解析参数、启动多 GPU worker
→ worker
→ load_joint_model
→ Qwen3ASRJointModel.transcribe
→ encode_offline / encode_stream / encode_train_mask
→ decode_aux（CTC/RNNT）和 decode_llm
→ joint.infer.merge：合并结果和 detail jsonl
```

主要从 [joint/infer.py](joint/infer.py) 的 `main → worker` 开始，再进入 [joint/model.py](joint/model.py) 的 `transcribe`。三种 `encoder_mode` 只在 `transcribe` 中分流，后续解码共用结果。若 checkpoint 没有 `joint_config.json`，`worker` 会走原始 `Qwen3ASRModel.transcribe` 分支，只支持离线 LLM。

### 3. 热词检索、注入与评测

热词推理不是独立模型入口，而是 Joint 推理中的一个分支：

```text
joint.infer.make_hotword
→ HotwordRetriever.from_file
→ Qwen3ASRJointModel.transcribe
→ 先用 CTC/RNNT 得到检索文本
→ HotwordRetriever.retrieve
→ defaults.hotword_prompt 把召回词写入 prompt
→ decode_llm 再识别一次
```

`HotwordRetriever.retrieve` 的内部路径是：

```text
phoneme.get_phoneme_info
→ retriever.FastRAG.search 粗筛
→ phoneme.fuzzy_substring_search_constrained DP 精筛
→ EnglishPhoneMatcher.retrieve 英文兜底
```

从 [hotword/retriever.py](hotword/retriever.py) 的 `HotwordRetriever.retrieve` 开始读最直接。`hotword/evaluate.sh` 先调用 Joint 推理生成 detail jsonl，再由 [hotword/evaluate.py](hotword/evaluate.py) 的 `evaluate → write_summary/write_badcases` 统计召回、误注入、修对和退化。

### 4. GRPO 热词强化学习

```text
grpo/train.sh
→ grpo.train.main
→ Qwen3ASRJointModel.from_pretrained(load_heads=False)
→ joint.lora.apply_lora
→ grpo.load_samples
→ RolloutSampler.sample
   → audio_embedding → encode_offline
   → _generate_batch
   → _logp_batch：冻结 old_logp 和 ref_logp
→ grpo.compute_reward
→ grpo.group_advantages
→ token_logp_batch：带梯度重算 current logp
→ grpo.grpo_loss：PPO clip + KL
→ 梯度同步、optimizer.step、save_checkpoint
```

先读 [grpo/train.py](grpo/train.py) 的 `main` 和 `RolloutSampler.sample`，再读 [grpo/grpo.py](grpo/grpo.py) 的 `compute_reward`、`group_advantages`、`grpo_loss`。LoRA 装配放在 [joint/lora.py](joint/lora.py)，因为 NLU 也复用它。

### 5. GLCLAP 训练与两种 benchmark

训练路径：

```text
glclap/train.sh
→ glclap.train.main
→ AudioTextDataset / GLCLAPCollator
→ GLCLAPModel.forward
→ encode_audio + encode_text
→ gather_embeddings
→ glclap_loss：global + local InfoNCE
→ evaluate / save_checkpoint
```

从 [glclap/train.py](glclap/train.py) 的 `main` 开始，再进入 [glclap/model.py](glclap/model.py) 的 `forward`。
版本训练参数与统一评测结果见 [glclap/README.md](glclap/README.md)。

两种评测共用 [glclap/benchmark.py](glclap/benchmark.py) 的数据读取和指标，但查询方式不同：

```text
GLCLAP：glclap.evaluate.main
→ load_hotword_benchmark(wav.scp)
→ encode_candidates
→ retrieve_batch（音频 embedding 对候选文本 embedding）
→ RetrievalMetrics

音素 baseline：glclap.phoneme_baseline.main
→ load_hotword_benchmark(text)
→ HotwordRetriever.retrieve（转写文本查热词）
→ RetrievalMetrics
```

因此 `evaluate.py` 衡量音频直接检索，`phoneme_baseline.py` 衡量已有转写文本上的音素检索，不是两份重复实现。

### 6. 文本 NLU / Agent

训练：

```text
nlu/scripts/train.sh
→ nlu.train.main
→ 按 backend 加载 Qwen3ASRJointModel 或 AutoModelForCausalLM
→ joint.lora.apply_lora 或普通 PEFT LoRA
→ NluCollator：渲染 prompt 并只保留 assistant labels
→ Trainer.train
```

`joint` backend 的 batch 没有音频，最终进入 `Qwen3ASRJointModel.forward(input_features=None)` 的纯文本 thinker 路径；`llm` backend 直接进入纯 Qwen3 LLM。

推理：

```text
nlu/scripts/infer.sh
→ nlu.infer.main
→ 多 GPU worker
→ build_nlu_prompt / tokenizer.apply_chat_template
→ qwen_model.generate / model.generate
→ merge
→ common.parse_intent 或 parse_agent
→ intent_metrics / agent_metrics + badcase
```

先读 [nlu/train.py](nlu/train.py) 或 [nlu/infer.py](nlu/infer.py) 的 `main`，格式解析和指标统一在 [nlu/common.py](nlu/common.py)。

### 7. ASR+NLU

训练：

```text
nlu/scripts/train_asr.sh
→ nlu.train_asr.main
→ expand_two_way：每条数据派生 ASR 与 ASR+NLU 两个样本
→ DataCollatorForJointTraining(need_aux=False)
→ joint.lora.apply_lora
→ Qwen3ASRJointModel.forward
→ encode_offline + thinker.forward，只计算 LLM loss
```

推理：

```text
nlu/scripts/infer_asr.sh
→ nlu.infer_asr.main
→ worker
→ joint.infer.load_joint_model
→ Qwen3ASRJointModel.transcribe(modes="llm")
→ split_text_intent
→ merge：文本 CER + intent_metrics
```

这条链路包含音频 collator 和 ASR 指标，因此与纯文本 NLU 分开。对应入口是 [nlu/train_asr.py](nlu/train_asr.py) 和 [nlu/infer_asr.py](nlu/infer_asr.py)。

### 8. 通用评测

[joint/scripts/infer.sh](joint/scripts/infer.sh) 在推理后继续调度：

- [evaluation/text_badcase.py](evaluation/text_badcase.py) → [evaluation/edit.py](evaluation/edit.py)：读取参考和预测，按文本 token 计算 WER 并输出 badcase；
- [evaluation/pinyin.py](evaluation/pinyin.py) → [evaluation/edit.py](evaluation/edit.py)：中英混合 token 化后计算拼音 PER/SAR 与 badcase；
- [hotword/evaluate.py](hotword/evaluate.py)：读取 Joint detail jsonl，计算热词召回和注入效果。

阅读评测代码时直接从各文件的 `main` 开始即可，它们不参与模型 forward。

## 常用入口

所有 Python 入口都从仓库根目录按模块运行，例如：

```bash
python3 -m qwen_asr_ext.joint.infer --help
python3 -m qwen_asr_ext.joint.train --help
python3 -m qwen_asr_ext.grpo.train --help
python3 -m qwen_asr_ext.glclap.evaluate --help
python3 -m qwen_asr_ext.glclap.phoneme_baseline --help
python3 -m qwen_asr_ext.glclap.train --help
python3 -m qwen_asr_ext.hotword.evaluate --help
python3 -m qwen_asr_ext.nlu.train --help
python3 -m qwen_asr_ext.nlu.infer --help
python3 -m qwen_asr_ext.nlu.train_asr --help
python3 -m qwen_asr_ext.nlu.infer_asr --help
python3 -m qwen_asr_ext.evaluation.pinyin --help
python3 -m qwen_asr_ext.evaluation.text_badcase --help
```

长期使用的参数配置仍保留在对应 `.sh` 文件中。目录只有一个 shell 时不额外创建 `scripts/`。

## GLCLAP benchmark 数据协议

`glclap/evaluate.py` 和 `glclap/phoneme_baseline.py` 共用 `benchmark.py`，并不限定 STOP1/STOP2。任意数据集只要整理成以下四个文件即可复用：

- `wav.scp`：`utt_id audio_path`，供 GLCLAP 音频查询；
- `text`：`utt_id transcript`，供音素基线和明细展示；
- `hotword.txt`：每行一个候选热词；
- `utt_hotword.txt`：`utt_id target_hotword`。

## 测试

根目录 `tests/` 只保留四组长期单元测试：GLCLAP 数据/损失、GRPO 数学、热词检索和热词奖励；`tests/conftest.py` 只负责让源码态 pytest 找到仓库包。它们验证共享协议与核心数学，不依赖个人 checkpoint，也不承担一次性数据检查。
