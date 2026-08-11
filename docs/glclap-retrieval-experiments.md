# GLCLAP 检索实验记录

更新时间：2026-08-11

## 1. 公共设置

| 项目 | 设置 |
|---|---|
| 训练集 | /cfs/data/private/WangYaoChi/train_data/all/train_700w_shuffled.jsonl |
| 验证集 | /cfs/data/private/WangYaoChi/train_data/all/eval_shuffled.jsonl |
| Audio Encoder | Data2Vec Audio Large |
| Text Encoder | bert-base-multilingual-uncased |
| 投影维度 | Audio/Text 分别经过 Linear 投影到 512 维 |
| global loss | 完整 transcript 与整段音频 embedding 的双向 InfoNCE |
| local loss | 连续 subtext 与音频帧 embedding 的双向 InfoNCE，时间维取 max |
| 总 loss | global loss + local loss |
| 音频 | 16 kHz mono，保留 0.2～30 秒 |
| 精度 | bf16 |
| Optimizer | AdamW，weight decay 0.01，grad clip 1.0 |
| 其他 | gradient checkpointing 开启，max text length 128，seed 42 |

全量微调模型共有 480,959,745 个可训练参数。V1 只训练两个投影层和 logit_scale；V2、V3 放开 Data2Vec、mBERT、投影层和 logit_scale。

## 2. 训练版本

### 2.1 主要参数

| 版本 | 模型目录 / checkpoint | 训练权重 | GPU | per-GPU / global batch | 实际 step / 计划 step |
|---|---|---|---:|---:|---:|
| V1 | /cfs/data/private/WangYaoChi/model/glclap/retrieval/checkpoint-47000 | 仅投影层和 logit_scale | 2 | 32 / 64 | 47k / 100k |
| V2 | /cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune/final | 全量微调 | 2 | 32 / 64 | 100k / 100k |
| V3 | /cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune_subtext_v2/checkpoint-160000 | 全量微调 | 8 | 8 / 64 | 160k / 200k |

### 2.2 优化参数与 subtext 构造

| 版本 | projection LR | Encoder LR | warmup | eval / save | subtext 构造 |
|---|---:|---:|---:|---:|---|
| V1 | 5e-4 | Encoder 冻结 | 1k | 1k / 1k | 中英文统一随机连续 2～8 units |
| V2 | 1e-3 | 1e-5 | 1k | 1k / 5k | 中英文统一随机连续 2～8 units |
| V3 | 1e-3 | 1e-5 | 10k | 10k / 10k | 中文 2～8；英文 1～4，权重 2:3:3:1 |

补充说明：

- V1 没有训练到计划的 100k，最后完整 checkpoint 为 47k；效果较差，不补跑统一测试。
- V2 共处理约 6.4M 个训练样本实例。
- V3 在 step 160k 时共处理约 10.24M 个训练样本实例。
- V3 英文 1/2/3/4 词的采样概率约为 22.22%/33.33%/33.33%/11.11%。
- V2 训练结束记录：step 100k train loss 0.1720；eval loss 0.0826，global R@1 99.84%，local R@1 97.34%，local R@5 99.92%。
- V2 与 V3 同为 global batch 64，但 V3 同时改变了 subtext 采样、训练步数和 warmup，因此两版差异不是严格的单变量实验。

## 3. 统一评测口径

| 数据集 | 评测目标数 | 候选数 | 说明 |
|---|---:|---:|---|
| STOP1 | 3,413 | 689 | 每条一个目标 |
| STOP2 | 7,112 | 1,061 | 每条一个目标 |
| AISHELL hotword | 281 | 185 | 232 条有标注音频；多目标按目标词展开，计算 micro recall |
| Voyah 联系人 | 10,004 | 2,000 | 从 11,322 条音频中保留可唯一匹配 contact_name 的样本 |

GLCLAP 评测：

- batch size 1，bf16。
- 固定返回 Top-3，不设置阈值，因此候选数大于等于 3 时每条一定返回 3 个。
- 在线延迟不包含 librosa 磁盘解码；包含 feature extraction、音频 Encoder 和候选相似度计算。

CTC 拼音/音素检索评测：

- 输入是 joint_ctc_50_grpo_2 产生的 ctc_text。
- 检索器为 HotwordRetriever：FastRAG 粗筛 + 边界约束 DP 精筛。
- 推理参数的 topk 上限为 3，但带分数阈值，因此每条实际返回 0～3 个候选，不会补满。
- 所以现有 hotword_eval.txt 中 R@5、R@10 与 R@3 相同。
- 这里的检索延迟只统计拿到 ctc_text 之后的 HotwordRetriever，不包含 CTC Encoder 和 CTC 解码时间；与 GLCLAP 在线延迟不是严格端到端同口径。

## 4. Top-1 / Top-3 对比

| 方法 | STOP1 | STOP2 | AISHELL hotword | Voyah 联系人 |
|---|---:|---:|---:|---:|
| CTC 粗识别 + 拼音/音素检索 | 60.42% / 75.51% | 83.41% / 86.14% | 75.80% / 94.66% | 98.08% / 98.94% |
| GLCLAP V2 | 49.63% / 63.40% | 29.16% / 37.09% | 71.89% / 92.53% | 85.80% / 95.48% |
| GLCLAP V3 checkpoint-160000 | 69.32% / 85.38% | 66.84% / 84.25% | 79.00% / 96.80% | 96.07% / 99.42% |

表格中每个单元格均为 Top-1 / Top-3。

STOP 按目标长度拆分：

| 数据集 | 分组 | 样本数 | V2 Top-1 / Top-3 | V3 Top-1 / Top-3 |
|---|---|---:|---:|---:|
| STOP1 | 1 unit | 1,141 | 5.26% / 12.71% | 45.31% / 72.13% |
| STOP1 | 多 unit | 2,272 | 71.92% / 88.86% | 81.38% / 92.03% |
| STOP2 | 1 unit | 4,716 | 5.26% / 10.05% | 60.60% / 80.15% |
| STOP2 | 多 unit | 2,396 | 76.21% / 90.32% | 79.13% / 92.32% |

## 5. 检索延迟

| 方法 | STOP1 mean | STOP2 mean | AISHELL mean | Voyah mean |
|---|---:|---:|---:|---:|
| CTC 拼音/音素检索，不含 CTC 推理 | 3.784 ms | 6.959 ms | 2.620 ms | 14.069 ms |
| GLCLAP V2 | 20.03 ms | 19.43 ms | 26.88 ms | 19.81 ms |
| GLCLAP V3 checkpoint-160000 | 18.32 ms | 18.27 ms | 27.51 ms | 19.43 ms |

## 6. 当前结论

- V3 在四个测试集上都明显优于 V2，说明增加短文本训练有效。
- 提升主要来自单 unit：STOP1 Top-1 从 5.26% 提升到 45.31%，STOP2 从 5.26% 提升到 60.60%。
- V3 在 STOP1 和 AISHELL 的 Top-1/Top-3 高于 CTC 拼音/音素检索。
- STOP2 上 CTC 检索仍明显更强；Voyah 上 CTC Top-1 更高，V3 Top-3 略高。
- 当前 CTC 延迟不含 CTC 模型推理，只能说明检索器本身较快，不能直接得出端到端比 GLCLAP 快。

## 7. 结果位置

| 方法 | 结果根目录 |
|---|---|
| CTC 拼音/音素检索 | /cfs/data/private/WangYaoChi/test_out/joint_ctc_50_grpo_2/asr_hotword |
| GLCLAP V2 | /cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune |
| GLCLAP V3 | /cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune_subtext_v2 |

CTC 四个子目录分别为 stop1、stop2、aishell_hotword、voyah；每个目录的 hotword_eval.txt 是汇总，details/results_detail.jsonl 是逐条检索明细。
