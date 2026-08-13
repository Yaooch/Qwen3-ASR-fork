# GLCLAP 检索实验记录

更新时间：2026-08-13

## 公共设置

| 项目 | 设置 |
|---|---|
| 训练 / 验证数据 | `train_700w_shuffled.jsonl` / `eval_shuffled.jsonl` |
| 模型 | Data2Vec Audio Large + multilingual BERT |
| 投影 | Audio / Text 分别经过 Linear 投影到 512 维 |
| 损失 | 双向 global InfoNCE + 双向 local InfoNCE；local 在音频时间维取 max |
| 音频 | 16 kHz mono，0.2～30 秒 |
| 优化 | AdamW，projection LR `1e-3`，Encoder LR `1e-5`，weight decay `0.01`，bf16 |

## 训练版本

| 版本 | 训练范围 | GPU × per-GPU batch | warmup | subtext 构造 | 当前 checkpoint |
|---|---|---:|---:|---|---:|
| V1 | 仅两个投影层和温度参数 | 2 × 32 | 1k | 中英文统一连续 2～8 units | 47k |
| V2 | Audio / Text Encoder 全量微调 | 2 × 32 | 1k | 中英文统一连续 2～8 units | 100k，训练完成 |
| V3 | Audio / Text Encoder 全量微调 | 8 × 8 | 10k | 中文 2～8；英文 1～4，概率 2:3:3:1 | 160k |
| V4 | Audio / Text Encoder 全量微调 | 7 × 9 | 10k | 中文 2～8，2/3 字加权；英文 1～4，概率 4:3:2:1；单词优先内容词和低频词 | 130k |

V4 的英文单词候选排除常见 stopword，要求训练集文档频次 `df>=5`；抽到单词时有 80% 概率从这些候选中按 `sqrt(median_df/df)` 加权采样。训练在 step 134330 连接断开，最后完整保存点为 `checkpoint-130000`。

## 统一评测结果

每个单元格为 Top-1 / Top-3。GLCLAP 固定返回 3 个候选，不设阈值；CTC 音素检索有阈值，实际返回 0～3 个。

| 方法 | STOP1 | STOP2 | AISHELL hotword | Voyah 联系人 |
|---|---:|---:|---:|---:|
| CTC 粗识别 + 音素检索 | 60.42% / 75.51% | 83.41% / 86.14% | 75.80% / 94.66% | 98.08% / 98.94% |
| GLCLAP V2 100k | 49.63% / 63.40% | 29.16% / 37.09% | 71.89% / 92.53% | 85.80% / 95.48% |
| GLCLAP V3 130k | 64.20% / 81.95% | 64.52% / 82.03% | 78.65% / 96.44% | 94.98% / 99.06% |
| GLCLAP V3 160k | 69.32% / 85.38% | 66.84% / 84.25% | 79.00% / 96.80% | 96.07% / 99.42% |
| GLCLAP V4 130k | **70.11% / 85.47%** | **67.28% / 84.76%** | 76.51% / 95.02% | 94.49% / 99.07% |

V3 与 V4 的 130k 同步对比：

| 数据集 | 分组 | 样本数 | V3 Top-1 / Top-3 | V4 Top-1 / Top-3 |
|---|---|---:|---:|---:|
| STOP1 | 1 unit | 1,141 | 39.26% / 67.84% | **50.66% / 72.30%** |
| STOP1 | 多 unit | 2,272 | 76.72% / 89.04% | **79.89% / 92.08%** |
| STOP2 | 1 unit | 4,716 | 57.53% / 77.08% | **60.22% / 81.17%** |
| STOP2 | 多 unit | 2,396 | 78.30% / 91.78% | **81.18% / 91.82%** |

V4 在同一步数显著改善 STOP，尤其是单 unit；但 AISHELL Top-1 下降 2.14 个百分点，Voyah Top-1 下降 0.49 个百分点，说明偏向英文短词的采样对中文专名存在轻微回退。

## 单条检索延迟

单位为 ms；GLCLAP 是 batch size 1 的音频检索时间，不含磁盘解码。CTC 行只统计取得 `ctc_text` 后的音素检索，不含 Qwen Encoder 和 CTC 解码，因此不是严格的端到端对比。

| 方法 | STOP1 mean | STOP2 mean | AISHELL mean | Voyah mean |
|---|---:|---:|---:|---:|
| CTC 音素检索 | 3.784 | 6.959 | 2.620 | 14.069 |
| GLCLAP V3 160k | 18.32 | 18.27 | 27.51 | 19.43 |
| GLCLAP V4 130k | 20.27 | 19.84 | 27.46 | 19.71 |

## 结果位置

- V2：`/cfs/data/private/WangYaoChi/model/glclap/retrieval_v2`
- V3：`/cfs/data/private/WangYaoChi/model/glclap/retrieval_v3`
- V4：`/cfs/data/private/WangYaoChi/model/glclap/retrieval_v4`
- CTC：`/cfs/data/private/WangYaoChi/test_out/joint_ctc_50_grpo_2/asr_hotword`

V3 / V4 的统一评测明细位于各版本目录下的 `eval_{stop1,stop2,aishell,voyah}_checkpoint-*_top3_latency.jsonl`。

## 下一实验：复用 Qwen3-ASR Encoder

目标是在最终 ASR 链路中复用已经计算的 Qwen3-ASR `ln_post` 特征，不再运行额外的 Data2Vec Encoder。文本侧继续使用 multilingual BERT，候选词 embedding 可离线预计算。

V5 用官方 Qwen3-ASR-1.7B 初始化，在 `train_mask` 下全量训练从 Mel 到 `ln_post` 的 Audio Encoder、mBERT、两个 512 维投影层和温度参数；Qwen `proj1/act/proj2` 不在检索计算图中。损失和 V4 数据采样保持不变，global batch 固定为 64（8 卡 × 8 或 4 卡 × 16）。本阶段只探索检索性能上限；若有效，再联合训练 CTC、LLM 和 GLCLAP。

V5 评测把完整 `ln_post` Encoder 输出作为计时边界：主延迟只统计 1024→512 投影、候选相似度和 Top-K，不包含特征提取与 Qwen Encoder。候选文本 embedding 仍离线预计算。
