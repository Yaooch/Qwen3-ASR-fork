# 用 GRPO 强化学习提升 LLM 热词选择能力

- 基线分支：`new-rag`
- 工作分支：`rl/hotword-grpo`
- 日期：2026-06-25
- 状态：待批准

## 目标与边界

热词两阶段里，检索（stage 1）已在 `new-rag` 上跑通。本期聚焦 stage 2 的 LLM 生成：通过 GRPO 强化学习，让 Qwen3-ASR 的 talker 在被注入「真热词 + 干扰词」混合列表时，**选对并原样输出真正被说到的热词，不误注入误召回的干扰词，同时非热词部分不退化**。

训练数据用 ContextASR（`/cfs/data/private/WangYaoChi/train_data/all/contextasr/train_contextasr2.jsonl`），每条已带 ground-truth 转写与混合热词列表，奖励可精确计算。

## 设计决策

- **RL 算法**：GRPO。组内相对优势，对可验证奖励友好，免训 value 模型。排除 PPO（需 critic、显存大、多模态 rollout 适配重）与 DPO（偏好微调非 on-policy，泛化边界弱，留作后续兜底）。
- **可训练范围**：仅 LoRA 打在 talker（thinker 文本解码器），冻结 audio_tower / 音频侧 / CTC / RNNT / lm_head / 原始权重。表达力够、非热词退化风险最低、显存最小。
- **候选列表来源**：用数据 prompt 里预给的热词列表（已是「真热词 + 干扰词」混合），与「给定列表让 LLM 选」目标完全对齐。retriever 是独立 stage 1，RL 不与之耦合；仿真差留到评测验证。
- **奖励真相**：从 `text` 字段取 ground-truth 转写，真热词 = 列表中 verbatim 出现在转写里的，干扰词 = 其余。
- **不套 trl**：trl GRPOTrainer 假设纯文本 LLM；本项目 Qwen3-ASR 是音频条件多模态，自写轻量 GRPO loop，复用现有 prompt 构造与采样路径。
- **YAGNI**：不做 retriever-in-rollout、DPO、全参 FT、KL 之外的 reward shaping。

## 奖励函数

每条样本定义：

- `G` = 真实转写（`parse_asr_output` 剥掉 `language X<asr_text>` 前缀后的纯文本，去标点归一）
- `H` = prompt 里的热词列表（去标点归一）
- `T = {h ∈ H : h 是 G 的子串}`（真热词，verbatim）
- `D = H \ T`（干扰词 / 误召回）
- `O` = 模型 rollout 原始输出（同样 `parse_asr_output` + 去标点）

三个分量：

1. **正确注入召回** `recall = |{h∈T : h 是 O 的子串}| / max(1, |T|)`
2. **误注入惩罚** `fp = |{h∈D : h 是 O 的子串}| / max(1, |D|)`
3. **非热词退化** `cer_nh = CER( mask_H(O), mask_T(G) )` —— 把 O 里所有 H 的子串抠掉、G 里所有 T 的子串抠掉再算 CER，隔离非热词质量，避免与 1/2 双重计分。
4. **格式兜底** `fmt ∈ {0,1}`：输出是否为合法转写（非空、未泄漏列表格式、未崩成乱码），防退化空串钻空子。

合成奖励：

```
R = w_r·recall − w_f·fp − w_c·cer_nh − w_fmt·(1−fmt)
```

默认权重 `w_r=1.0, w_f=1.0, w_c=0.5, w_fmt=0.5`（可调）。边界：`|T|=0` 时 recall 项置 0（仅靠 `−w_f·fp − w_c·cer_nh` 约束，鼓励「没人说到就别注入」）；`max(1,·)` 防除零。

GRPO 用法：每条样本采样 G=8 个 rollout，各自算 `R`，组内 `(R−μ)/σ` 归一化作优势 `A`。

## GRPO 训练循环与数据流

**数据准备（离线一次性）**：

1. 读 `train_contextasr2.jsonl`，每条解析 `audio`、`G`（parse 后纯转写）、`H`（正则从 `prompt` 抠 `专属名词：[...]`）。
2. 预计算 `T`/`D`、去标点 `G_text`、去标点热词，缓存为训练样本（避免每步重算）。
3. 切 train / eval（eval 不参与梯度，仅做 reward 监控）。

**Rollout（每个 step、每条样本采样 G=8）**：

1. 加载音频 → feature extractor → audio_tower（frozen）→ 音频 embedding（同一条音频 8 个 rollout 共享，只算一次）。
2. 用数据预给 `prompt`（含热词列表）按现有 `_build_text_prompt` 拼 LLM 输入（含 audio placeholder），LoRA talker 带温度采样生成 8 条 `O`。
3. 每条 `O` 算第 1 节奖励 `R`，组内归一化得优势 `A`。

**梯度更新**：

- 对每条 rollout 重算 token-level logprob（LoRA talker 前向），GRPO 目标：
  `L = −mean[ A·logπ_θ(y|x) ] + β·KL(π_θ ‖ π_ref)`
- `π_ref` = 冻结基线 talker（LoRA merge 关闭即得，零额外显存），`β=0.04`。
- 仅更新 LoRA 参数；AdamW，lr ~1e-5~2e-5（LoRA 量级），cosine schedule。
- bf16 + LoRA + 音频 embedding 缓存 + gradient checkpointing，目标单机可跑。

## LoRA 目标与冻结范围

- **可训练（LoRA）**：`Qwen3ASRForConditionalGeneration.thinker` 的 `Qwen3ASRThinkerTextModel` 注意力 `q/k/v/o_proj` 与 MLP `gate/up/down_proj`。配置 `r=16, alpha=32, dropout=0.05`，bf16。
- **冻结**：audio_tower、thinker 音频侧、CTC head、RNNT head、lm_head、thinker 文本解码器除 LoRA 外原始权重，`requires_grad_(False)`，不进 optimizer。
- `lm_head` 不动：词表大、全参风险高；热词选择靠表示质量，lm_head 冻结已够。

## 评测与成功标准

**评测集**：RL 专用 eval split + 现有 `hotword_eval` 数据集（用真实 retriever 列表验仿真差）。

**指标**：

1. 热词召回（正确注入）：`|T∩O|/|T|`，越高越好。
2. 误注入率：`|D∩O|/|D|`，越低越好。
3. 非热词 CER：去标点非热词部分，对比 RL 前基线，**不退化**为红线。
4. 整体 CER：兜底总览。

**成功阈值**：热词召回相对基线提升、误注入率下降、非热词 CER 不劣化（≤ 基线 +0.5% 绝对），三者同时达成才算通过。

## 分支、范围与风险

- **分支**：从 `new-rag` 新建 `rl/hotword-grpo`。
- **代码位置**：训练脚本放 `finetuning/`，奖励/匹配工具放 `qwen_asr/tools/`，不动现有 `transcribe` 推理路径；RL 训完 LoRA 可选 merge 或挂载进 `infer`。
- **风险**：
  1. 多模态 rollout 自写 loop 工程量 > 套 trl —— 主要工作量集中于此。
  2. verbatim 子串匹配对中文同形/标点边界有噪声 —— 去标点 + 缓存一次性算好，可接受。
  3. 仿真差：数据预给列表干扰词分布 ≠ 真实 retriever 误召回分布 —— 靠真实 retriever 评测把关，必要时小批量 retriever 列表补训（v2，不在本期）。

## 不在本期范围

- retriever-in-rollout（训练时跑真实 retriever 生成列表）
- DPO / 全参 FT
- KL 之外的 reward shaping（如 length penalty、过程奖励）
- 训后 LoRA merge 进基线权重的流程
