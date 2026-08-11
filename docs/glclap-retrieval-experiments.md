# GLCLAP 检索训练实验记录

更新时间：2026-08-11

## 1. 目标与固定口径

当前只复现论文的音频到文本候选检索部分，不接入 Qwen3-ASR。主要追踪 V2 全量微调与 V3 短文本增强全量微调；V1 仅训练投影层，效果明显较差，不再补跑完整测试。

固定评测设置：

- 推理 dtype：bf16。
- batch size：1。
- 输出固定 Top-3，不设分数阈值，因此候选数不少于 3 时每条一定返回 3 个结果。
- 同时报告 Top-1 recall 和 Top-3 recall。
- 候选文本 embedding 预先离线计算，单独统计 candidate encode 时间。
- 在线单条延迟不含 librosa 磁盘解码；包含 feature extraction、CPU 到 GPU 拷贝、音频 Encoder、与全部候选的相似度计算和 topk。
- 明细由 finetuning/eval_glclap.py 生成。

## 2. 模型与 loss

模型代码：qwen_asr/joint/glclap.py。

- Audio Encoder：Data2Vec Audio Large。
- Text Encoder：bert-base-multilingual-uncased。
- Audio Encoder 输出经过无 bias Linear 投影到 512 维。
- Text Encoder 输出经过无 bias Linear 投影到 512 维。
- 可训练温度参数 logit_scale，初始温度 0.07。
- Audio global embedding：有效音频帧投影后的 masked mean，再做 L2 normalize。
- Audio local embedding：每帧投影后分别做 L2 normalize。
- Text embedding：BERT token hidden state 的 masked mean，经投影后 L2 normalize。
- 总参数及全量微调参数：480,959,745。
  - Data2Vec Audio：313,276,416。
  - mBERT：166,765,824。
  - Audio projection：524,288。
  - Text projection：393,216。
  - logit_scale：1。

训练同时使用两项对称 InfoNCE：

- global：完整 transcript text embedding 与 audio global embedding。
- local：随机连续 subtext embedding 与所有 audio frame 比较，对时间维取 max。
- 每项都计算 text-to-audio 和 audio-to-text 两个方向并取平均。
- 总 loss = global_loss + local_loss。

当前 local loss 的正例仍严格使用 batch 对角线；相同 subtext 出现在同一 global batch 时会形成假负例。当前推理和 local loss 都使用单帧 max pooling。

## 3. 训练数据公共设置

- train：/cfs/data/private/WangYaoChi/train_data/all/train_700w_shuffled.jsonl
- eval：/cfs/data/private/WangYaoChi/train_data/all/eval_shuffled.jsonl
- JSONL 只有 audio 和 text 字段，没有实体类型标注。
- text 格式：language X<asr_text>文本。
- 音频统一按 16 kHz、mono 读取。
- 有效时长：0.2 到 30 秒。
- IterableDataset 按 rank 和 worker 做字节分片，读完后循环。
- 完整 transcript 用于 global loss；从 transcript 抽取的连续 span 用于 local loss。
- optimizer：AdamW。
- dtype：bf16。
- gradient checkpointing：开启。
- grad clip：1.0。
- weight decay：0.01。
- seed：42。
- max text length：128。
- 每 rank 4 个 DataLoader worker。

## 4. V2：全量微调基线

模型目录：

/cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune/final

对应训练代码基线 commit：c8194e6。

训练设置：

- GPU：4、5，共 2 张 A800。
- per-GPU batch：32。
- global batch：64。
- max steps：100,000。
- 约处理 6.4M 个样本实例。
- projection 与 logit_scale LR：1e-3。
- Data2Vec 和 mBERT LR：1e-5。
- warmup：1,000 steps。
- scheduler：cosine，最终 LR 降到 0。
- eval every：1,000 steps，每次 20 batches。
- save every：5,000 steps。
- Data2Vec：全部放开。
- mBERT：全部放开。
- subtext：中英文均从 2 到 min(8, 句长) 均匀选择长度，再随机选连续 span；不足 2 units 时使用全文。
- 这意味着英文单词不会以单词级目标参与 local loss，除非整句本身只有一个 unit。

训练结束记录：

- step 100,000：loss 0.1720，global 0.0007，local 0.2481。
- eval：loss 0.0826，global R@1 0.9984，local R@1 0.9734，local R@5 0.9992。

## 5. V3：短文本增强全量微调

评测 checkpoint：

/cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune_subtext_v2/checkpoint-160000

对应采样代码 commit：543979b。

训练设置：

- GPU：0 到 7，共 8 张 A800。
- per-GPU batch：8。
- global batch：64，与 V2 相同。
- 计划 max steps：200,000；当前用于比较的是 step 160,000。
- step 160,000 已处理约 10.24M 个样本实例；计划完成时为 12.8M。
- projection 与 logit_scale LR：1e-3。
- Data2Vec 和 mBERT LR：1e-5。
- warmup：10,000 steps。
- scheduler：cosine。
- eval every：10,000 steps。
- save every：10,000 steps。
- Data2Vec、mBERT：全部放开。
- 中文 subtext 长度 2 到 8，权重为 3:3:1:1:1:1:1。
- 英文 subtext 长度 1 到 4，权重为 2:3:3:1，实际目标概率为 22.22%、33.33%、33.33%、11.11%。
- 短句不能支持全部长度时，在可用长度内重新归一化。

注意：V3 与 V2 不只是采样策略不同，训练步数和 warmup 也不同。因此结果差异不能全部归因于单词采样；要做因果结论，需要相同初始化、步数和 scheduler 的单变量实验。

## 6. 测试集口径

### STOP1 / STOP2

- STOP1：3,413 条，689 个去重候选。
- STOP2：7,112 条，1,061 个去重候选。
- 每条一个目标热词。

### AISHELL hotword test

路径：

/cfs/data/private/WangYaoChi/open_datasets/aishell_hotword_test

- 235 条 wav.scp，其中 232 条有非空热词标注。
- 一条音频可包含多个目标，分布为：191 条含 1 个、36 条含 2 个、2 条含 3 个、3 条含 4 个。
- 共 281 个目标实例。
- hotword.txt 原始 187 行，normalize 和去重后为 185 个候选。
- 评测时按逗号或中文逗号拆开目标，将一条多目标音频展开成多个 audio-target 实例。
- 主表 Top-1/Top-3 是 281 个目标实例的 micro recall。
- 另报告按原音频聚合的 top-k 任一目标命中与全部目标命中。

### Voyah 联系人

路径：

/cfs/data/private/hubk/asr_test_set/VOYAH_CONTACT_TEST_SET

- contact_name：2,000 个唯一联系人，作为候选库。
- contact_wavs：11,322 条。
- 用联系人名字是否为 wav 文件名的唯一子串生成目标。
- 10,004 条可唯一映射到一个联系人，作为联系人检索评测集。
- 1,318 条“呼叫一二七”等数字拨号音频不含联系人目标，排除。
- 没有多联系人歧义样本。

## 7. 统一 Top-1 / Top-3 结果

| 数据集 | V2 Top-1 | V2 Top-3 | V3 Top-1 | V3 Top-3 | V3 Top-1 增益 |
|---|---:|---:|---:|---:|---:|
| STOP1 | 49.63% | 63.40% | 69.32% | 85.38% | +19.69 |
| STOP2 | 29.16% | 37.09% | 66.84% | 84.25% | +37.68 |
| AISHELL，目标词 micro | 71.89% | 92.53% | 79.00% | 96.80% | +7.11 |
| Voyah 联系人 | 85.80% | 95.48% | 96.07% | 99.42% | +10.27 |

STOP 按目标长度拆分：

| 数据集 | 分组 | 样本数 | V2 Top-1 | V2 Top-3 | V3 Top-1 | V3 Top-3 |
|---|---|---:|---:|---:|---:|---:|
| STOP1 | 1 unit | 1,141 | 5.26% | 12.71% | 45.31% | 72.13% |
| STOP1 | 多 unit | 2,272 | 71.92% | 88.86% | 81.38% | 92.03% |
| STOP2 | 1 unit | 4,716 | 5.26% | 10.05% | 60.60% | 80.15% |
| STOP2 | 多 unit | 2,396 | 76.21% | 90.32% | 79.13% | 92.32% |

AISHELL 按 232 条原音频聚合：

| 模型 | Top-1 任一命中 | Top-3 任一命中 | Top-3 全部命中 |
|---|---:|---:|---:|
| V2 | 87.07% | 96.12% | 91.38% |
| V3 | 95.69% | 98.28% | 96.12% |

结果表明 V3 的主要收益确实集中在单 unit：STOP1 Top-1 提升 40.05 点，STOP2 提升 55.34 点；多 unit 的提升分别只有 9.46 和 2.92 点。AISHELL 与 Voyah 也同时提升，说明收益不是只出现在 STOP。但训练步数是混杂变量，仍需单变量实验确认。

## 8. 延迟

| 数据集 | 模型 | candidate encode | mean | p50 | p95 | max |
|---|---|---:|---:|---:|---:|---:|
| STOP1 | V2 | 519.28 ms | 20.03 ms | 19.24 ms | 28.15 ms | 50.91 ms |
| STOP1 | V3 | 约 526 ms | 18.32 ms | 17.49 ms | 25.33 ms | 33.35 ms |
| STOP2 | V2 | 588.56 ms | 19.43 ms | 18.92 ms | 23.72 ms | 42.40 ms |
| STOP2 | V3 | 约 567 ms | 18.27 ms | 18.07 ms | 20.53 ms | 34.97 ms |
| AISHELL | V2 | 522.47 ms | 26.88 ms | 28.77 ms | 30.42 ms | 34.30 ms |
| AISHELL | V3 | 448.26 ms | 27.51 ms | 29.37 ms | 30.89 ms | 42.28 ms |
| Voyah | V2 | 708.71 ms | 19.81 ms | 19.64 ms | 23.58 ms | 45.30 ms |
| Voyah | V3 | 684.20 ms | 19.43 ms | 19.01 ms | 22.90 ms | 54.34 ms |

不同任务音频时长不同，因此不能跨数据集直接比较 mean。此次多个任务并行运行，CPU feature extraction 可能受并发影响；这些数字适合判断量级，不作为严格 GPU kernel benchmark。

## 9. 下一版数据策略定义

### 9.1 英文长度比例

下一版目标比例固定为：

- 1 词：40%。
- 2 词：30%。
- 3 词：20%。
- 4 词：10%。

先采样长度，再在该长度的连续 span 中选择目标。句长小于采样长度时，只在可用长度内按原比例重新归一化。

### 9.2 stopword 的含义

这里的 stopword 是语言学上的高频功能词，例如 the、a、to、in、is、of，不是 STOP1/STOP2 数据集。

它只用于控制英文 1 词样本的质量，不删除 STOP 数据，也不应完全禁止功能词。建议 1 词样本中：

- 80% 从非 stopword 内容词池采样。
- 20% 从全部词均匀采样，保留 the、play 等真实高频命令覆盖。
- 如果一句话只有 stopword，则回退到全部词池。

### 9.3 低频词

使用训练集 document frequency，而不是测试集频率：

df(s) = 包含 normalized span s 的训练 utterance 数。

按语言和 span 长度分别统计，避免把中文 2 字词和英文单词放在同一分布里。初始可操作定义：

- df 小于 5 的候选先视为可能的标注噪声，不做强增采样。
- df 不小于 5，且位于同语言、同长度候选 df 分布的后 50%，标记为 low-frequency。
- 不做硬二选一时，采样权重使用 inverse-sqrt，并限制最大倍率：
  rarity_weight = clamp(sqrt(median_df / max(df, 5)), 0.5, 4.0)。

最终阈值要在训练集频率统计完成后固定，不能使用 STOP、AISHELL 或 Voyah 测试结果反推。

### 9.4 专有名词

当前 JSONL 只有 audio/text，没有实体元数据，无法从字段直接得到专有名词。推荐离线标注训练 transcript：

- 中文：NER 的 PER、LOC、ORG，配合独立的训练词典。
- 英文：NER 的 PERSON、GPE/LOC、ORG；原始大小写可作辅助特征，但不能单独作为定义。
- 实体 span 必须是 transcript 的连续原文片段。
- 只采用高置信度实体，并记录 entity_type。
- entity span 的采样权重初始乘 2，再通过 dev set 调整。
- 禁止把 STOP、AISHELL hotword.txt 或 Voyah contact_name 加入训练实体词典，否则会测试泄漏。

如果暂时不引入 NER，先使用“非 stopword + 训练集低频内容词”作为专有名词代理，比仅依赖英文首字母大写稳定。

## 10. hard negative 的实操方案

先从训练 transcript 建立独立候选词典，保存 text、language、unit length、df、phoneme、entity_type。然后为每个正例离线缓存最多 4 个 hard negatives：

1. 拼写相近：相同语言、长度相近，按字符编辑相似度或 char n-gram 召回。
2. 音素相近：中文用无声调拼音或音素，英文用 gruut 音素，按 normalized edit distance 召回。
3. 同类型实体：PER 对 PER、LOC 对 LOC、ORG 对 ORG，同时约束长度接近。
4. 同前缀或后缀：共享首尾字符、wordpiece 或首尾音素，但完整文本不同。

约束：

- 排除与正例完全相同的 normalized text。
- 发音完全相同的异形词不要作为普通 hard negative；音频本身可能不可区分，应排除或视作 pronunciation-equivalent positive。
- hard negative 只能来自训练侧词典，不能来自测试候选库。
- 相似度不能只取最相似的极端样本；每类从一个相似度区间随机采样，避免全部是标签歧义。
- 缓存不足时回退到普通 in-batch negatives。

在当前 local loss 上增加与真实推理方向一致的 audio-to-candidate 辅助 CE。对每条音频计算正 subtext 与 K 个 hard-negative text 的 frame-max score：

hard_loss = CE([score(audio, positive), score(audio, neg_1), ..., score(audio, neg_K)], label=0)

建议：

- 总 loss = global_loss + local_multi_positive_loss + lambda_hard * hard_loss。
- lambda_hard 从 0 线性升到 0.5，前 10,000 steps 完成 ramp。
- 首轮 K=4，每类最多 1 个，先观察显存和收敛。
- 单独报告 hard-negative Top-1 accuracy 和正负 score margin。

## 11. 假负例的确认与修复

先测量，不直接猜测。每 100 steps 记录 global batch 内：

- exact normalized subtext 的重复组数。
- 至少与另一条重复的样本比例。
- 按 1/2/3/4 词分别统计 collision rate。
- stopword 与非 stopword 分开统计。
- 重复组大小的 p50/p95/max。

DDP 下不能只去除单卡重复，因为 global batch 跨 8 卡。建议把 normalized subtext 生成稳定 64-bit hash，跨 rank gather 后构造 positive mask。

推荐修复是 multi-positive contrastive loss：同一 normalized subtext 对应的所有音频都算 local positive。对每一行使用正例集合的 logsumexp 减去全候选 logsumexp；转置方向同样计算。这样不会为了避免重复而破坏多卡 IterableDataset，也保留高频真实词的训练数据。

“同 batch 不重复”可作为对照实验，但跨 rank 去重实现更复杂，而且会系统性降低高频词曝光；不建议作为最终主方案。

## 12. 单帧 max pooling 的诊断

当前 score(text, audio) 是所有有效帧相似度的最大值。可能的问题有两类：

- 一个噪声帧形成偶然高分，尤其长音频帧数更多时负例极值更大。
- 一个词跨多个音素，单帧峰值不能稳定表达完整词。

先在同一个 checkpoint 上做不训练的 pooling ablation：

- max。
- top-3 / top-5 frame mean。
- 3 / 5 / 9 帧滑动窗口 mean 后再取 max。
- length-normalized logsumexp。

对每种 pooling 报告：

- 四个测试集 Top-1/Top-3。
- 1 unit 与多 unit 分组。
- 按音频时长分桶后的 recall 和正负 margin。
- 正例峰值连续宽度，以及错误 Top-1 的峰值宽度。
- 负例最大分数与音频帧数的相关性。

判断标准：

- 如果长音频的负例 max 明显升高，且 window/top-k pooling 改善长音频，说明存在极值偏置。
- 如果正确词通常只有 1 帧尖峰，错误候选也以尖峰取胜，而连续窗口改善单词 Top-1，说明单帧证据不稳定。
- 如果无训练替换 pooling 无提升，只能说明现有表示适配 max；仍可用相同初始化做短程 pooling-loss 对照，不能直接断言问题不存在。

## 13. 推荐实验顺序

为了能解释每个改动，下一轮不要一次加入全部机制：

1. 只加统计：subtext collision、长度实际分布、词频分布；模型不变。
2. 推理侧 pooling ablation；模型不变。
3. E1 数据采样：英文 40/30/20/10，加非 stopword 与低频/实体加权；loss 不变。
4. E2 在 E1 上只加入 multi-positive local loss。
5. E3 在 E2 上只加入 K=4 hard-negative auxiliary loss。
6. 只有 pooling 诊断为阳性时，再做 E4 pooling-loss 训练对照。

筛选阶段可都从 V3 checkpoint-160000 初始化，使用新 optimizer 做约 20k steps 的低学习率短跑，以节省成本；最终确认版必须从相同预训练基座、相同 step、相同 scheduler 完整训练一次。建议短跑起始 LR：projection 1e-4、encoder 2e-6、warmup 1,000，具体值先以 loss/margin 稳定性为准。

STOP1、STOP2、AISHELL、Voyah 应作为最终测试集，不反复用于阈值和采样倍率调参。调参使用 eval_shuffled 中独立冻结的 dev 子集，并额外构建不与四个测试候选重叠的 entity dev set。

## 14. 结果文件

V2：

- /cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune/eval_stop1_final_top3_latency.jsonl
- /cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune/eval_stop2_final_top3_latency.jsonl
- /cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune/eval_aishell_final_top3_latency.jsonl
- /cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune/eval_voyah_final_top3_latency.jsonl

V3：

- /cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune_subtext_v2/eval_stop1_checkpoint-160000_top3_latency.jsonl
- /cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune_subtext_v2/eval_stop2_checkpoint-160000_top3_latency.jsonl
- /cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune_subtext_v2/eval_aishell_checkpoint-160000_top3_latency.jsonl
- /cfs/data/private/WangYaoChi/model/glclap/retrieval_full_finetune_subtext_v2/eval_voyah_checkpoint-160000_top3_latency.jsonl
