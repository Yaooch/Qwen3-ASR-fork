# 复现 asr-hotword 检索并接入现有热词流程

- 分支：`new-rag`
- 日期：2026-06-23
- 状态：已批准

## 目标与边界

复现 [asr-hotword](https://github.com/HaujetZhao/asr-hotword) 的两层检索（粗筛 `FastRAG` + 精筛边界约束 DP），**只取检索召回**，丢弃其替换/黑名单/去重逻辑。召回词按本项目现有方式注入 LLM prompt，不做文本替换。评测记录每条音频的检索耗时与召回，精度指标沿用现有 `hotword_eval.py`。两套 retriever（`pinyin` / `asr_hotword`）经开关切换，同一数据集分别跑，产出可比的精度+耗时对比。

## 设计决策

- **对方代码放置**：子包保真拷贝。把对方 4 个核心检索文件拷进 `qwen_asr/joint/asr_hotword/`，只做最小适配（去 `logger` 依赖、删 `__main__` 测试与未接入的冗余实现）。算法常量与逻辑逐字保留，保证复现保真。子包内保留对方代码风格。
- **召回阈值策略**：按精筛最高分降序取 topk，不加额外阈值下限，保证 topk 填满、与 `pinyin` retriever 公平对比（命中少于 topk 时返回全部）。
- **代码风格**：尽可能简洁、干净，复现基本功能即可，不做额外抽象与配置。

## 子包结构 `qwen_asr/joint/asr_hotword/`

| 文件 | 来源 | 改动 |
|------|------|------|
| `phoneme.py` | `algo_phoneme.py`（`Phoneme`、`get_phoneme_info`） | 去 `logger` import、删 `__main__` |
| `calc.py` | `algo_calc.py`（`SIMILAR_PHONEMES`、`fuzzy_substring_search_constrained`、`get_phoneme_cost`） | 基本不动（本身不依赖 logger） |
| `fast_rag.py` | 合并 `rag_fast.py`（`PhonemeEncoder`）+ `rag_fast_batch.py`（`FastRAG`） | 去 logger 调用、删 `__main__`/对比测试 |
| `retriever.py` | 新写 | `AsrHotwordRetriever` adapter |
| `__init__.py` | 新写 | 导出 `AsrHotwordRetriever` |

丢弃：`hot_phoneme.py`（替换/黑名单）、`rag_fast_rf.py`、`rag_accu.py`、`benchmark.py`、`gen_test_data.py`。

## AsrHotwordRetriever 接口

与 `HotwordRetriever` 对齐：

```python
class AsrHotwordRetriever:
    @classmethod
    def from_file(cls, path, **kwargs) -> "AsrHotwordRetriever": ...
    def retrieve(self, query: str, topk: int = 10) -> List[str]: ...
```

`retrieve` 内部流程：

1. `get_phoneme_info(query)` → query 音素序列（带位置/边界）
2. `FastRAG.search(query_phonemes)` → 粗筛 `[(hw, score, approx_end_pos)]`
3. 精筛编排（复用对方 `_find_matches` 思路）：按 target 聚合位置、位置去重（距离 <5 合并）、在 `window = approx_end ± (hw_len+10)` 内跑 `fuzzy_substring_search_constrained` → 每个 hw 取所有位置的最高精筛分
4. 按最高分降序取 topk

热词加载：本项目热词文件每行一个词、无别名，故 `{word: [[phonemes]]}` 每词一条音素序列。超参（粗筛/精筛阈值）经 `__init__` 暴露，默认值对齐对方。

## 接入点改动

- `finetuning/infer.py`：新增 `--hotword_retriever` `choices=[pinyin, asr_hotword]` 默认 `pinyin`；`make_hotword()` 据此选 `HotwordRetriever` 或 `AsrHotwordRetriever`。
- `qwen_asr/joint/model.py`（transcribe，293-303）：`retrieve` 前后 `time.perf_counter()` 打点，写 `rec["hotword_retrieve_ms"]`（仅当有 retriever 时）。该字段随 rec 自动进 detail jsonl。
- `qwen_asr/joint/__init__.py`：导出 `AsrHotwordRetriever`。
- `finetuning/hotword_eval.sh`：透传 `--hotword_retriever`（可选）。

## 耗时统计

`qwen_asr/tools/hotword_eval.py` 读 detail 每条 `hotword_retrieve_ms`，统计 `mean / p50 / p95 / max / 总耗时 / 有效样本数`，summary 新增「检索耗时」小节。per-utt 耗时已落盘 detail jsonl，可后续分析分布。

## 评测与对比流程

同一数据集分别用 `--hotword_retriever pinyin` 和 `asr_hotword` 各跑一遍 `infer + eval`，得两份 summary，对比 R@k、召回准确率、误召回、识别率、检索耗时。badcase 复用现有分组。

## 依赖与风险

- 依赖 `pypinyin`、`rapidfuzz`，本项目均已具备，无新增。
- 对方精筛是纯 Python DP，大词表+长文本可能慢——耗时统计正是要暴露的点。
- 复现保真：不动对方算法常量；如要调参另作。
