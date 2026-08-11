# 开发说明

Qwen3-ASR 仓库包含官方 ASR 包、推理/服务入口，以及联合 CTC/RNNT 流式实验代码。实验环境：`conda activate qwen3-asr`。

## 目录边界

- 官方能力放在 `qwen_asr/core/`、`qwen_asr/inference/`、`qwen_asr/cli/`。
- 联合 CTC/RNNT 实验核心放在 `qwen_asr/joint/`：`model.py` 负责训练 forward 和 `transcribe` 主推理入口，`encoder.py` 负责 Encoder、长度换算和流式 KV cache，`ctc.py/rnnt.py/defaults.py` 分别放对应核心逻辑。
- 热词检索只保留 `qwen_asr/joint/hotword/` 的音素级两层实现：`FastRAG` 粗筛 + 边界约束 DP 精筛，统一由 `HotwordRetriever` 对外提供接口。
- `finetuning/` 只放长期使用的训练、推理、评估入口；`qwen_asr/tools/` 只保留主链路依赖或会重复使用的评估/诊断工具，一次性数据加工脚本不要提交到仓库。
- 默认 prompt、训练词表路径、训练 SentencePiece 路径、WER 脚本路径和流式常量统一放在 `qwen_asr/joint/defaults.py`。

## 代码原则

- 保持最小修改：一两行能解决的直接在原逻辑处写，不为单次简单逻辑新增 helper、兼容开关或历史实验开关。
- 复用现有 helper：训练/推理入口优先复用 `defaults.py`、`model.py` 的配置解析 helper 和 `encoder.py` 的长度/流式 helper，不把核心模型逻辑塞进入口脚本。
- 不再通过 Mixin/MRO 扩展 joint 推理入口；`Qwen3ASRJointModel.transcribe` 是主入口。
- 函数名尽量短，必要注释用中文；打印输出保持简洁中文。
- 只在改动较大、实现与本文档不符，或确有必要时更新 `AGENTS.md`；普通 debug 或几行修复不需要同步改文档。
- 仓库根目录不放训练/评测 JSONL；本项目 NLU 数据统一放在 `/cfs/data/private/WangYaoChi/train_data/all/nlu/`，生成报告放在被忽略的 `reports/` 或 CFS。

## 联合训练约定

- `--train` 只接受 `llm/proj/encoder/ctc/rnnt` 的逗号组合；`proj` 对应 `audio_tower.proj1/act/proj2`，不再使用 `train_mode/aux_loss_type`。
- CTC head 固定叫 `ctc`，RNNT 固定 cached greedy 解码；两者新训练固定接在 `ln_post` 后、`proj1/proj2` 前。
- CTC adapter 支持 `mlp/moe`，`joint_config.json` 记录 `ctc_adapter`，旧 checkpoint 默认 `mlp`。
- CTC-only 训练不构造 LLM `input_ids/labels`；训练 batch 不携带未使用的 raw waveform；checkpoint 保存统一走 `JointTrainer.save_model`，输出根目录和 `checkpoint-*` 都应可直接推理。
- 训练验证指标记录 `eval_ctc_cer/eval_rnnt_cer`，TensorBoard 日志路径由 `finetuning/train.py --logging_dir` 控制。

## NLU / Agent 训练约定

- 纯文本 NLU/Agent 统一使用 `finetuning/train_nlu.py` 和 `finetuning/infer_nlu.py`，通过 `--backend joint/llm` 显式区分 joint checkpoint 与纯 Qwen3 LLM；不再新增 `*_nlu_pure.py/sh` 平行入口。
- 两种文本 backend 共用 JSONL 读取、label mask、结果合并、指标和 badcase；仅模型/Tokenizer 加载、prompt 渲染和 LoRA target 保留必要分支。
- `joint` 无音频训练走 `Qwen3ASRJointModel.forward(input_features=None)` 的 thinker 文本路径；`llm` 直接走 `AutoModelForCausalLM`。
- ASR+NLU 因为包含音频 collator、ASR/ASR+NLU 双样本派生与 CER 评测，继续保留 `train_asr_nlu.py/infer_asr_nlu.py` 独立入口，不并入文本 NLU。
- `--resume_from` 先加载 LoRA；只有目录含 `trainer_state.json` 时才同时恢复 Trainer optimizer/step。

## 流式和推理约定

- 联合训练和批量推理默认使用 `DEFAULT_ATTN_IMPLEMENTATION="flash_attention_2"`；未安装 flash-attn 时不要盲目调大 encoder batch。
- `--stream_train 1` 使用训练侧 WeNet-like 帧级 chunk mask：默认左看 24 帧、当前 6 帧、右看 2 帧；若同时训练 LLM，复用该流式 encoder 输出再过 `proj1/act/proj2`。
- 批量推理用 `--encoder_mode offline/stream/train_mask`；`train_mask` 是训练侧 chunk mask 理论上限，不代表线上真实流式行为；兼容保留 `--stream/--no_stream` 作为别名。
- `--mode` 只接受 `llm/ctc/rnnt` 的逗号组合，不再使用 `joint` 模式。
- 真实流式推理每 640ms 处理一个 waveform chunk，batch 内 active chunk 合批送 encoder KV cache；CTC/RNNT 流式解码固定 batched greedy。
- 多 GPU 批量推理由子进程自行读取 scp 并分片，避免通过启动 pipe 传递大 shard。

## 热词和评估约定

- 热词召回固定使用 `HotwordRetriever` 的音素级两层检索，不再提供算法选择或拼音风格参数；缺少 `pypinyin/rapidfuzz` 时直接报错。
- 热词推理可用 `--keep_origin_llm 0/1` 控制是否保留默认 LLM 输出，默认保留；热词库和评估目标热词分开，评估目标格式为 `utt_id<TAB>热词1,热词2`。
- 热词测试集按语言拆分输出到测试集目录下的 `Mandarin/` 和 `English/`，`hotword.txt` 从拆分后的目标热词重新生成。
- 拼音评估兼容 `utt_id<TAB>文本` 和 `utt_id 空格 文本`；中英混合文本保留 ASCII token，汉字转拼音后一起计算 PER/SAR。
- WER 阶段同时生成文本 badcase，输出 `utt_id/WER/ref/hyp`；badcase 保留问题分组标题，单条只输出 `utt_id/target/retrieved/ref/llm/final`。
- 启用热词检索时，`Qwen3ASRJointModel.transcribe` 每条记录 `hotword_retrieve_ms`（毫秒）写入 detail jsonl，`qwen_asr/tools/hotword_eval.py` 汇总 mean/p50/p95/max。

## 热词 GRPO 强化学习

- 目标：让 talker 在注入「真热词 + 干扰词」混合列表时选对并原样输出真热词、不误注入干扰词、非热词部分不退化。
- 训练数据用 ContextASR（`train_contextasr2.jsonl`，每条 `text` 为 ground-truth、`prompt` 已含混合热词列表）；候选列表直接用数据预给列表，不与 retriever 耦合。
- 奖励纯函数在 `qwen_asr/tools/hotword_reward.py`：`parse_text_field` 剥 `language X<asr_text>` 前缀并去标点，`split_truth` 按子串划真热词/干扰词，`compute_reward = w_r·recall − w_f·fp − w_mix·hybrid_injection − w_c·non_hotword_cer − w_fmt·(1−fmt)`；`hybrid_injection_rate` 惩罚真热词与干扰词片段拼出的中文混合词。
- 训练组件在 `finetuning/grpo_core.py`（数据读写/组内优势+clip+KL 数学/LoRA 装配，轻量、可单测）与 `finetuning/grpo_train.py`（`RolloutSampler` 音频 embedding + G 路采样 + ref/train logprob + 训练主循环）；评测统一走 `finetuning/hotword_eval.sh` + `qwen_asr/tools/hotword_eval.py`。
- 模型加载统一 `from_pretrained(..., load_heads=False, attn_implementation=...).to("cuda")`（GRPO 只用 LLM + audio_tower，不加载 CTC/RNNT 头）；LoRA 评测经 `PeftModel.from_pretrained(joint, lora)` 包整个 joint，与训练的 adapter key 对齐。
- 训练 `bash finetuning/grpo_train.sh [OUTPUT_DIR] [NPROC] [RESUME]`（默认 4 卡，经 `torchrun` 数据并行：各卡取 rank-strided 样本做 G 路 rollout 与 backward，LoRA 梯度 all-reduce 求平均后同步步进；`apply_lora` 后由 rank 0 广播 LoRA 初始参数，保证各卡一致），单卡跑传 `NPROC=1`；`--batch_size_per_rank` 控制每卡每步累积样本数，effective batch = NPROC × batch_size_per_rank；`RESUME=1` 从 `OUTPUT_DIR/lora` 续训（恢复 LoRA 权重 + 优化器状态 + step）。评测 `bash finetuning/hotword_eval.sh`（设 `lora` 变量挂载 RL LoRA，留空评基线）。成功标准：热词召回升、误注入降、非热词 CER 不劣化（≤基线 +0.5% 绝对），三者同时达成。
- 纯函数与数学单测在 `tests/tools/test_hotword_reward.py`、`tests/finetuning/test_grpo_core.py`；不保留依赖个人 checkpoint 的集成测试。

## 脚本和验证

- Shell 参数解析避免为每个 `--arg` 手写重复 `case` 分支，优先使用通用赋值 helper，只保留布尔开关和特殊副作用分支。
- 一次性数据切分、扰动和统计脚本放在 CFS 或临时目录，不放入 `qwen_asr/tools/`；大数据统计优先随机 seek 抽样，避免全量扫描。
- 修改 Python 文件后，至少运行 `python3 -m py_compile <file...>`。
- 修改 shell 脚本后，至少运行 `bash -n <file...>`。
- 如果改动影响共享接口、训练/推理主链路或跨模块行为，再按实际影响范围扩大验证。
