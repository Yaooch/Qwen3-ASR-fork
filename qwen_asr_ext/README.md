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
    ├── pinyin.py           # 拼音 PER/SAR 与 badcase
    └── text_badcase.py     # 文本 WER badcase
```

核心调用关系是：`joint` 提供模型与编码主链路；`hotword` 只负责检索；`grpo` 训练 joint 的文本解码器 LoRA；`glclap` 是独立检索实验；`nlu` 复用 joint 模型和 LoRA 装配。LoRA 放在 `joint/lora.py`，因为 GRPO、文本 NLU 和 ASR+NLU 都会使用它。

## 常用入口

所有 Python 入口都从仓库根目录按模块运行，例如：

```bash
python3 -m qwen_asr_ext.joint.infer --help
python3 -m qwen_asr_ext.joint.train --help
python3 -m qwen_asr_ext.grpo.train --help
python3 -m qwen_asr_ext.glclap.evaluate --help
python3 -m qwen_asr_ext.glclap.phoneme_baseline --help
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
