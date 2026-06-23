# asr-hotword 检索复现接入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use sdd:subagent-driven-development (recommended) or sdd:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `new-rag` 分支复现 asr-hotword 的两层检索（粗筛 FastRAG + 精筛边界约束 DP），以 adapter 接入现有「召回→注入 LLM prompt」流程，评测记录每条音频检索耗时与召回，两套 retriever 经开关切换对比。

**Architecture:** 新建子包 `qwen_asr/joint/asr_hotword/`，保真移植对方 4 个核心检索文件（去掉 logger/__main__/未接入冗余），新增 `AsrHotwordRetriever` adapter 实现 `retrieve(query, topk)->List[str]` 接口。`infer.py` 加 `--hotword_retriever` 开关，`model.py` 在 retrieve 前后打点写 `hotword_retrieve_ms`，`hotword_eval.py` 汇总检索耗时。后续处理（注入 prompt、LLM 二次解码）完全复用现有逻辑，不做文本替换。

**Tech Stack:** Python 3.9+，pypinyin 0.55.0，rapidfuzz 3.14.5（均已安装，无新增依赖）。

## Global Constraints

- 分支：`new-rag`。每个 task 末尾 commit，一个 commit 一个逻辑改动，commit message 用中文。
- 算法常量保真：`SIMILAR_PHONEMES`、DP 代价 0.5、粗筛阈值 0.55、召回阈值 0.65 不改。
- 代码简洁：只移植检索必需函数，删 `__main__` 测试块、logger、未接入的 `PhonemeIndex`/`rag_accu`/`hot_phoneme` 替换逻辑。
- 项目无 pytest，验证用内联 `python -c`，在项目根运行（`PYTHONPATH=.`）。
- 移植自 `/tmp/asr-hotword/hotword/`（对方 MIT 许可）。

## File Structure

| 文件 | 责任 | 新建/修改 |
|------|------|-----------|
| `qwen_asr/joint/asr_hotword/__init__.py` | 导出 `AsrHotwordRetriever` | 新建 |
| `qwen_asr/joint/asr_hotword/phoneme.py` | 文本→带位置/边界音素序列（`Phoneme`、`get_phoneme_info`） | 新建 |
| `qwen_asr/joint/asr_hotword/calc.py` | 精筛 DP（`SIMILAR_PHONEMES`、`fuzzy_substring_search_constrained`） | 新建 |
| `qwen_asr/joint/asr_hotword/fast_rag.py` | 粗筛（`PhonemeEncoder`、`FastRAG`） | 新建 |
| `qwen_asr/joint/asr_hotword/retriever.py` | adapter `AsrHotwordRetriever` | 新建 |
| `qwen_asr/joint/__init__.py` | 顶层包导出 `AsrHotwordRetriever` | 修改 |
| `finetuning/infer.py` | `--hotword_retriever` 开关 + `make_hotword` 分流 | 修改 |
| `qwen_asr/joint/model.py` | retrieve 打点写 `hotword_retrieve_ms` | 修改 |
| `qwen_asr/tools/hotword_eval.py` | 汇总检索耗时（mean/p50/p95/max） | 修改 |
| `finetuning/hotword_eval.sh` | 透传 `--hotword_retriever` | 修改 |

---

## Task 1: 移植 phoneme.py（音素转换）

**Files:**
- Create: `qwen_asr/joint/asr_hotword/__init__.py`（空包初始化）
- Create: `qwen_asr/joint/asr_hotword/phoneme.py`

**Interfaces:**
- Produces: `Phoneme`（dataclass，`info` 属性返回七元组）、`get_phoneme_info(text)->List[Phoneme]`

- [ ] **Step 1: 验证当前状态（应失败：模块不存在）**

Run: `PYTHONPATH=. python3 -c "from qwen_asr.joint.asr_hotword.phoneme import get_phoneme_info; get_phoneme_info('撒贝宁')"`
Expected: `ModuleNotFoundError`

- [ ] **Step 2: 创建空包 `__init__.py`**

文件 `qwen_asr/joint/asr_hotword/__init__.py` 内容（本 task 仅占位，Task 4 填导出）：

```python
"""asr-hotword 两层检索复现：粗筛 FastRAG + 精筛边界约束 DP。"""
```

- [ ] **Step 3: 创建 `phoneme.py`（移植自对方 algo_phoneme.py，去 logger/__main__/未用函数）**

文件 `qwen_asr/joint/asr_hotword/phoneme.py` 完整内容：

```python
# coding: utf-8
"""音素转换：文本到带位置/边界的音素序列。移植自 asr-hotword。"""
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple, Literal

from pypinyin import pinyin, Style


@dataclass(frozen=True)
class Phoneme:
    value: str
    lang: Literal['zh', 'en', 'num', 'other']
    is_word_start: bool = False
    is_word_end: bool = False
    char_start: int = 0
    char_end: int = 0

    @property
    def is_tone(self) -> bool:
        return self.value.isdigit()

    @property
    def info(self) -> Tuple[str, str, bool, bool, bool, int, int]:
        return (self.value, self.lang, self.is_word_start, self.is_word_end,
                self.is_tone, self.char_start, self.char_end)


def get_phoneme_info(text: str, ascii_split_char: bool = True) -> List[Phoneme]:
    phoneme_seq: List[Phoneme] = []
    pos = 0
    while pos < len(text):
        char = text[pos]
        if '一' <= char <= '鿿':
            pos = _process_zh(text, pos, phoneme_seq)
        elif 'a' <= char.lower() <= 'z' or '0' <= char <= '9':
            pos = _process_en_num(text, pos, phoneme_seq, ascii_split_char)
        else:
            cat = unicodedata.category(char)
            if cat.startswith('L'):
                phoneme_seq.append(Phoneme(
                    char.lower(), 'other',
                    is_word_start=True, is_word_end=True,
                    char_start=pos, char_end=pos + 1,
                ))
            pos += 1
    return phoneme_seq


def _process_zh(text: str, pos: int, seq: List[Phoneme]) -> int:
    zh_start = pos
    scan_pos = pos + 1
    while scan_pos < len(text) and '一' <= text[scan_pos] <= '鿿':
        scan_pos += 1
    zh_end = scan_pos
    fragment = text[zh_start:zh_end]
    try:
        py_initials = pinyin(fragment, style=Style.INITIALS, strict=False, errors='ignore')
        py_finals = pinyin(fragment, style=Style.FINALS, strict=False, errors='ignore')
        py_tones = pinyin(fragment, style=Style.TONE3, neutral_tone_with_five=True, errors='ignore')
        min_len = min(len(fragment), len(py_initials), len(py_finals), len(py_tones))
        for i in range(min_len):
            idx = zh_start + i
            init, fin, tone = py_initials[i][0], py_finals[i][0], py_tones[i][0]
            items = []
            if init:
                items.append(Phoneme(init, 'zh', is_word_start=True, char_start=idx, char_end=idx + 1))
            if fin:
                items.append(Phoneme(fin, 'zh', is_word_start=not init, char_start=idx, char_end=idx + 1))
            if tone and tone[-1].isdigit():
                items.append(Phoneme(tone[-1], 'zh', is_word_end=True, char_start=idx, char_end=idx + 1))
            if not items:
                items.append(Phoneme(fragment[i], 'zh', is_word_start=True, is_word_end=True,
                                     char_start=idx, char_end=idx + 1))
            seq.extend(items)
    except Exception:
        for i, c in enumerate(fragment):
            seq.append(Phoneme(c, 'zh', is_word_start=True, is_word_end=True,
                               char_start=zh_start + i, char_end=zh_start + i + 1))
    return zh_end


def _process_en_num(text: str, pos: int, seq: List[Phoneme], split_char: bool) -> int:
    start_pos = pos
    while pos < len(text):
        char = text[pos]
        low_char = char.lower()
        if not ('a' <= low_char <= 'z' or '0' <= char <= '9'):
            break
        if pos > start_pos:
            prev = text[pos - 1]
            if (prev.islower() and char.isupper()) or \
               (prev.isalpha() and char.isdigit()) or \
               (prev.isdigit() and char.isalpha()):
                break
        pos += 1
    end_pos = pos
    token = text[start_pos:end_pos].lower()
    lang = 'num' if token.isdigit() else 'en'
    if split_char:
        for i, c in enumerate(token):
            seq.append(Phoneme(c, lang, is_word_start=(i == 0), is_word_end=(i == len(token) - 1),
                               char_start=start_pos + i, char_end=start_pos + i + 1))
    else:
        seq.append(Phoneme(token, lang, is_word_start=True, is_word_end=True,
                           char_start=start_pos, char_end=end_pos))
    return end_pos
```

- [ ] **Step 4: 验证音素转换正确（三字 → 三个字起点）**

Run: `PYTHONPATH=. python3 -c "from qwen_asr.joint.asr_hotword.phoneme import get_phoneme_info; ps=get_phoneme_info('撒贝宁'); assert len(ps)>=6 and sum(p.is_word_start for p in ps)==3, [p.value for p in ps]; print('OK', [p.value for p in ps])"`
Expected: `OK ['s', 'a', '1', 'b', 'ei', '4', 'n', 'ing', '2']`（声调数字可能因 pypinyin 版本略有不同，断言只验三个字起点）

- [ ] **Step 5: Commit**

```bash
git add qwen_asr/joint/asr_hotword/__init__.py qwen_asr/joint/asr_hotword/phoneme.py
git commit -m "移植 asr-hotword 音素转换 phoneme.py"
```

---

## Task 2: 移植 calc.py（精筛 DP）

**Files:**
- Create: `qwen_asr/joint/asr_hotword/calc.py`

**Interfaces:**
- Consumes: `from .phoneme import Phoneme`
- Produces: `SIMILAR_PHONEMES`、`fuzzy_substring_search_constrained(hw_info, input_info, threshold)->List[(score, start_idx, end_idx)]`

- [ ] **Step 1: 验证当前状态（应失败）**

Run: `PYTHONPATH=. python3 -c "from qwen_asr.joint.asr_hotword.calc import fuzzy_substring_search_constrained"`
Expected: `ModuleNotFoundError`

- [ ] **Step 2: 创建 `calc.py`（移植自对方 algo_calc.py，import 路径改为本子包；删未接入的 `get_phoneme_cost`/`find_best_match`）**

文件 `qwen_asr/joint/asr_hotword/calc.py` 完整内容：

```python
# coding: utf-8
"""精筛：边界约束的模糊编辑距离 DP。移植自 asr-hotword。"""
from typing import List, Tuple

from .phoneme import Phoneme  # noqa: F401  （保留类型导出，调用方按 info 元组传入）


SIMILAR_PHONEMES = [
    {'an', 'ang'},
    {'en', 'eng'},
    {'in', 'ing'},
    {'ian', 'iang'},
    {'uan', 'uang'},
    {'z', 'zh'},
    {'c', 'ch'},
    {'s', 'sh'},
    {'l', 'n'},
    {'f', 'h'},
    {'ai', 'ei'},
    {'o', 'uo'},
    {'e', 'ie'},
    {'p', 't'},
    {'p', 'b'},
    {'t', 'd'},
    {'k', 'g'},
]


def _is_similar_phoneme(a: str, b: str) -> bool:
    pair = {a, b}
    return any(pair.issubset(s) for s in SIMILAR_PHONEMES)


def lcs_length(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    m, n = len(s1), len(s2)
    if n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


def fuzzy_substring_search_constrained(hw_info: List[Tuple], input_info: List[Tuple],
                                        threshold: float = 0.6) -> List[Tuple[float, int, int]]:
    n = len(hw_info)
    m = len(input_info)
    if n == 0 or m == 0:
        return []

    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    path = [[(0, 0)] * (m + 1) for _ in range(n + 1)]

    input_vals = [t[0] for t in input_info]
    input_langs = [t[1] for t in input_info]
    input_starts = [t[2] for t in input_info]
    hw_vals = [t[0] for t in hw_info]
    hw_langs = [t[1] for t in hw_info]
    hw_phones = [t[4] for t in hw_info]

    for j in range(m + 1):
        if j == 0 or (j < m and input_starts[j]):
            dp[0][j] = 0.0
            path[0][j] = (0, j)

    for i in range(1, n + 1):
        h_v, h_l, h_p = hw_vals[i - 1], hw_langs[i - 1], hw_phones[i - 1]
        row_min = float('inf')
        for j in range(1, m + 1):
            i_v, i_l = input_vals[j - 1], input_langs[j - 1]
            if h_l != i_l:
                cost = 1.0
            elif h_v == i_v:
                cost = 0.0
            elif h_l == 'zh':
                if h_p:
                    cost = 0.5
                elif _is_similar_phoneme(h_v, i_v):
                    cost = 0.5
                else:
                    cost = 1.0
            elif h_l == 'en':
                lcs = lcs_length(h_v, i_v)
                cost = 1.0 - (lcs / max(len(h_v), len(i_v)))
            else:
                cost = 1.0

            dist_match = dp[i - 1][j - 1] + cost
            dist_del = dp[i - 1][j] + 1.0
            dist_ins = dp[i][j - 1] + 1.0
            if dist_match <= dist_del:
                if dist_match <= dist_ins:
                    dp[i][j] = dist_match
                    path[i][j] = path[i - 1][j - 1]
                else:
                    dp[i][j] = dist_ins
                    path[i][j] = path[i][j - 1]
            else:
                if dist_del <= dist_ins:
                    dp[i][j] = dist_del
                    path[i][j] = path[i - 1][j]
                else:
                    dp[i][j] = dist_ins
                    path[i][j] = path[i][j - 1]
            if dp[i][j] < row_min:
                row_min = dp[i][j]
        if row_min > n * (1.0 - threshold) + 2:
            break

    results = []
    for j in range(1, m + 1):
        if not input_info[j - 1][3]:
            continue
        dist = dp[n][j]
        if dist >= n * 0.8:
            continue
        score = 1.0 - (dist / n)
        if score >= threshold:
            start_idx = path[n][j][1]
            results.append((score, start_idx, j))
    results.sort(key=lambda x: x[0], reverse=True)

    used_ends = {}
    for score, s, e in results:
        if e not in used_ends or score > used_ends[e][0]:
            used_ends[e] = (score, s, e)
    return sorted(used_ends.values(), key=lambda x: x[0], reverse=True)
```

- [ ] **Step 3: 验证精筛能定位子串（「贝宁」在「撒贝宁」中高分匹配）**

Run: `PYTHONPATH=. python3 -c "from qwen_asr.joint.asr_hotword.calc import fuzzy_substring_search_constrained as f; from qwen_asr.joint.asr_hotword.phoneme import get_phoneme_info; hw=[p.info[:5] for p in get_phoneme_info('贝宁')]; inp=[p.info for p in get_phoneme_info('撒贝宁')]; r=f(hw, inp, 0.6); assert r and r[0][0] > 0.8, r; print('OK', r)"`
Expected: `OK [(1.0, ...)]`（「贝宁」是「撒贝宁」子串，应近满分）

- [ ] **Step 4: Commit**

```bash
git add qwen_asr/joint/asr_hotword/calc.py
git commit -m "移植 asr-hotword 精筛 DP calc.py"
```

---

## Task 3: 移植 fast_rag.py（粗筛 FastRAG）

**Files:**
- Create: `qwen_asr/joint/asr_hotword/fast_rag.py`

**Interfaces:**
- Consumes: `from .phoneme import Phoneme`
- Produces: `PhonemeEncoder`、`FastRAG(threshold)`，`FastRAG.add_hotwords({word: [[Phoneme]]})`、`FastRAG.search(input_phonemes, top_k=0)->List[(hw, score, approx_end_pos)]`

- [ ] **Step 1: 验证当前状态（应失败）**

Run: `PYTHONPATH=. python3 -c "from qwen_asr.joint.asr_hotword.fast_rag import FastRAG"`
Expected: `ModuleNotFoundError`

- [ ] **Step 2: 创建 `fast_rag.py`（合并对方 rag_fast.py 的 PhonemeEncoder + rag_fast_batch.py 的 FastRAG，去计时/logger/__main__）**

文件 `qwen_asr/joint/asr_hotword/fast_rag.py` 完整内容：

```python
# coding: utf-8
"""FastRAG 粗筛：rapidfuzz 全局批量匹配 + 掩码剥离多位置召回。移植自 asr-hotword。"""
from typing import Dict, List, Tuple

import rapidfuzz.fuzz as _fuzz
import rapidfuzz.distance.OSA as _OSA
import rapidfuzz.process as _process

from .phoneme import Phoneme


class PhonemeEncoder:
    """将音素字符串编码为整数，加速比较效率。"""

    def __init__(self):
        self.phoneme_to_code: Dict[str, int] = {}
        self.code_to_phoneme: Dict[int, str] = {}
        self.next_code = 1  # 0 保留

    def encode(self, phoneme: str) -> int:
        if phoneme not in self.phoneme_to_code:
            self.phoneme_to_code[phoneme] = self.next_code
            self.code_to_phoneme[self.next_code] = phoneme
            self.next_code += 1
        return self.phoneme_to_code[phoneme]

    def encode_sequence(self, phonemes: List[str]) -> List[int]:
        return [self.encode(p) for p in phonemes]


class FastRAG:
    """rapidfuzz 全局批量加速版 RAG 检索器。"""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.encoder = PhonemeEncoder()
        # {(热词, 音素编码元组): 编码列表}
        self.hotwords: Dict[Tuple[str, Tuple[int, ...]], List[int]] = {}
        self.hotword_count = 0

    def add_hotwords(self, hotwords: Dict[str, List[List[Phoneme]]]) -> None:
        for hw, phoneme_lists in hotwords.items():
            for phonemes in phoneme_lists:
                if phonemes:
                    codes = self.encoder.encode_sequence([p.value for p in phonemes])
                    self.hotwords[(hw, tuple(codes))] = codes
                    self.hotword_count += 1

    def search(self, input_phonemes: List[Phoneme], top_k: int = 0) -> List[Tuple[str, float, int]]:
        """检索相关热词（top_k<=0 时返回全部）。返回 [(hw, score, approx_end_pos)]。"""
        if not input_phonemes or not self.hotwords:
            return []
        input_list = self.encoder.encode_sequence([p.value for p in input_phonemes])
        pr_cutoff = self.threshold * 100

        matches = _process.extract(
            input_list, self.hotwords, scorer=_fuzz.partial_ratio,
            score_cutoff=pr_cutoff, limit=None,
        )
        results = []
        for _match_val, _score, key in matches:
            hw, hw_tuple = key
            hw_list = list(hw_tuple)
            hw_len = len(hw_list)
            osa_cutoff = int(hw_len * (1 - self.threshold))

            remaining_input = list(input_list)
            while True:
                alignment = _fuzz.partial_ratio_alignment(remaining_input, hw_list, score_cutoff=pr_cutoff)
                if alignment is None:
                    break
                if any(remaining_input[idx] == -1 for idx in range(alignment.src_start, alignment.src_end)):
                    break
                aligned = input_list[alignment.src_start:alignment.src_end]
                dist = _OSA.distance(aligned, hw_list, score_cutoff=osa_cutoff)
                if dist <= osa_cutoff:
                    score = 1.0 - (dist / hw_len)
                    end_pos = alignment.src_start + hw_len
                    results.append((hw, round(score, 3), end_pos))
                for idx in range(alignment.src_start, alignment.src_end):
                    remaining_input[idx] = -1

        final = {}
        for hw, score, end_pos in results:
            key = (hw, end_pos)
            if key not in final or score > final[key][0]:
                final[key] = (score, end_pos)
        results = [(hw, score, end_pos) for (hw, _), (score, end_pos) in final.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results if top_k <= 0 else results[:top_k]
```

- [ ] **Step 3: 验证粗筛召回（「撒贝你主持」召回到「撒贝宁」）**

Run: `PYTHONPATH=. python3 -c "from qwen_asr.joint.asr_hotword.fast_rag import FastRAG; from qwen_asr.joint.asr_hotword.phoneme import get_phoneme_info; rag=FastRAG(0.55); rag.add_hotwords({'撒贝宁':[get_phoneme_info('撒贝宁')],'康辉':[get_phoneme_info('康辉')]}); r=rag.search(get_phoneme_info('撒贝你主持'), top_k=5); assert any(hw=='撒贝宁' for hw,_,_ in r), r; print('OK', r)"`
Expected: `OK [('撒贝宁', <score>, <pos>), ...]`

- [ ] **Step 4: Commit**

```bash
git add qwen_asr/joint/asr_hotword/fast_rag.py
git commit -m "移植 asr-hotword 粗筛 fast_rag.py"
```

---

## Task 4: 写 adapter AsrHotwordRetriever 并导出

**Files:**
- Create: `qwen_asr/joint/asr_hotword/retriever.py`
- Modify: `qwen_asr/joint/asr_hotword/__init__.py`（填导出）
- Modify: `qwen_asr/joint/__init__.py`（顶层导出）

**Interfaces:**
- Consumes: `get_phoneme_info`（phoneme）、`fuzzy_substring_search_constrained`（calc）、`FastRAG`（fast_rag）
- Produces: `AsrHotwordRetriever.from_file(path)->AsrHotwordRetriever`、`retrieve(query, topk=10)->List[str]`

- [ ] **Step 1: 验证当前状态（应失败）**

Run: `PYTHONPATH=. python3 -c "from qwen_asr.joint.asr_hotword import AsrHotwordRetriever"`
Expected: `ImportError`

- [ ] **Step 2: 创建 `retriever.py`（adapter，复用对方 _find_matches 精筛编排，只取召回分数）**

文件 `qwen_asr/joint/asr_hotword/retriever.py` 完整内容：

```python
# coding: utf-8
"""asr-hotword 检索 adapter：复现对方两层检索，仅产出召回词列表。"""
from typing import Dict, List, Optional

from .phoneme import Phoneme, get_phoneme_info
from .calc import fuzzy_substring_search_constrained
from .fast_rag import FastRAG


class AsrHotwordRetriever:
    """音素级两层检索（粗筛 FastRAG + 精筛边界约束 DP），接口对齐 HotwordRetriever。"""

    def __init__(
        self,
        hotwords: List[str],
        fast_threshold: float = 0.55,
        recall_threshold: float = 0.65,
    ):
        self.hotwords = [h.strip() for h in hotwords if h.strip()]
        self.fast_threshold = fast_threshold
        self.recall_threshold = recall_threshold
        self._phonemes: Dict[str, List[List[Phoneme]]] = {}
        self._rag = FastRAG(threshold=fast_threshold)
        for word in self.hotwords:
            phons = get_phoneme_info(word)
            if phons:
                self._phonemes[word] = [phons]
        self._rag.add_hotwords(self._phonemes)

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "AsrHotwordRetriever":
        with open(path, "r", encoding="utf-8") as f:
            hotwords = [line.strip() for line in f if line.strip()]
        return cls(hotwords, **kwargs)

    def retrieve(self, query: str, topk: int = 10) -> List[str]:
        if not self.hotwords or not query:
            return []
        input_phonemes = get_phoneme_info(query)
        if not input_phonemes:
            return []

        fast_results = self._rag.search(input_phonemes, top_k=0)
        # 精筛编排（复用对方 _find_matches 思路）：按 target 聚合位置、位置去重、窗口内跑 DP，取每词最高分。
        seen: Dict[str, List[int]] = {}
        for hw, _score, approx_end in fast_results:
            positions = seen.setdefault(hw, [])
            if not any(abs(approx_end - p) < 5 for p in positions):
                positions.append(approx_end)

        input_info = [p.info for p in input_phonemes]
        best: Dict[str, float] = {}
        for hw, positions in seen.items():
            for approx_end in positions:
                for hw_phonemes in self._phonemes.get(hw, []):
                    hw_compare = [p.info[:5] for p in hw_phonemes]
                    window_size = len(hw_compare) + 10
                    win_start = max(0, approx_end - window_size)
                    win_end = min(len(input_info), approx_end + 5)
                    local_input = input_info[win_start:win_end]
                    for score, _s, _e in fuzzy_substring_search_constrained(
                        hw_compare, local_input, threshold=self.fast_threshold
                    ):
                        if score > best.get(hw, 0.0):
                            best[hw] = score

        ranked = [(w, s) for w, s in best.items() if s >= self.recall_threshold]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in ranked[:topk]]
```

- [ ] **Step 3: 填子包 `__init__.py` 导出**

文件 `qwen_asr/joint/asr_hotword/__init__.py` 完整内容（覆盖 Task 1 的占位）：

```python
"""asr-hotword 两层检索复现：粗筛 FastRAG + 精筛边界约束 DP。"""
from .retriever import AsrHotwordRetriever

__all__ = ["AsrHotwordRetriever"]
```

- [ ] **Step 4: 顶层 `qwen_asr/joint/__init__.py` 增加导出**

在 `from .hotword import HotwordRetriever` 下一行加：

```python
from .asr_hotword import AsrHotwordRetriever
```

并在 `__all__` 列表中 `"HotwordRetriever",` 下一行加 `"AsrHotwordRetriever",`。

修改后 `qwen_asr/joint/__init__.py` 完整内容：

```python
"""Joint CTC/RNNT extensions for Qwen3-ASR."""

from .ctc import CTC, CTCAdapter, CTCMoEAdapter
from .hotword import HotwordRetriever
from .asr_hotword import AsrHotwordRetriever
from .model import Qwen3ASRJointModel
from .defaults import DEFAULT_PROMPT, JOINT_CONFIG, hotword_prompt
from .rnnt import RNNT

__all__ = [
    "CTC",
    "CTCAdapter",
    "CTCMoEAdapter",
    "HotwordRetriever",
    "AsrHotwordRetriever",
    "Qwen3ASRJointModel",
    "DEFAULT_PROMPT",
    "JOINT_CONFIG",
    "RNNT",
    "hotword_prompt",
]
```

- [ ] **Step 5: 端到端验证（recall「撒贝宁」并过阈值）**

Run: `PYTHONPATH=. python3 -c "from qwen_asr.joint import AsrHotwordRetriever as R; r=R(['撒贝宁','康辉','周涛','东方财富']); out=r.retrieve('撒贝你主持的节目', topk=3); assert '撒贝宁' in out, out; print('OK', out)"`
Expected: `OK ['撒贝宁']`（或含其他高分词，但「撒贝宁」必须在内）

- [ ] **Step 6: Commit**

```bash
git add qwen_asr/joint/asr_hotword/retriever.py qwen_asr/joint/asr_hotword/__init__.py qwen_asr/joint/__init__.py
git commit -m "新增 AsrHotwordRetriever adapter 并接入 joint 包导出"
```

---

## Task 5: infer.py 接入 retriever 切换开关

**Files:**
- Modify: `finetuning/infer.py`（parse_args 加参数；make_hotword 分流）

**Interfaces:**
- Consumes: `AsrHotwordRetriever`（lazy import）、现有 `HotwordRetriever`
- Produces: CLI 参数 `--hotword_retriever {pinyin,asr_hotword}` 默认 `pinyin`

- [ ] **Step 1: 加 CLI 参数**

在 `finetuning/infer.py` 的 `parse_args()` 中，`--hotword_pinyin_style` 那行（第 34 行）之后加一行：

```python
    parser.add_argument("--hotword_retriever", choices=["pinyin", "asr_hotword"], default="pinyin")
```

- [ ] **Step 2: 改 `make_hotword` 分流**

将 `finetuning/infer.py:68-73` 的 `make_hotword` 函数替换为：

```python
def make_hotword(args):
    if not args.hotword_file:
        return None
    if "llm" not in args.modes or not ({"ctc", "rnnt"} & set(args.modes)):
        raise ValueError("热词 prompt 需要同时跑 llm 和 ctc/rnnt，例如 --mode llm,ctc")
    if args.hotword_retriever == "asr_hotword":
        from qwen_asr.joint.asr_hotword import AsrHotwordRetriever
        return AsrHotwordRetriever.from_file(args.hotword_file)
    return HotwordRetriever.from_file(args.hotword_file, pinyin_style=args.hotword_pinyin_style)
```

- [ ] **Step 3: 验证参数生效**

Run: `PYTHONPATH=. python3 finetuning/infer.py --help | grep -A2 hotword_retriever`
Expected: 输出包含 `--hotword_retriever {pinyin,asr_hotword}` 且 default 为 `pinyin`

- [ ] **Step 4: 验证 asr_hotword 路径能加载热词（用真实热词文件，若不存在则跳过）**

Run: `PYTHONPATH=. python3 -c "from qwen_asr.joint.asr_hotword import AsrHotwordRetriever; r=AsrHotwordRetriever.from_file('/cfs/data/private/WangYaoChi/open_datasets/aishell_hotword_test/hotword.txt') if __import__('os').path.exists('/cfs/data/private/WangYaoChi/open_datasets/aishell_hotword_test/hotword.txt') else None; print('loaded', len(r.hotwords) if r else 'skip')" 2>&1 | tail -1`
Expected: `loaded <N>`（若热词文件存在）或 `skip`（不存在则跳过，不阻断）

- [ ] **Step 5: Commit**

```bash
git add finetuning/infer.py
git commit -m "infer.py 加 --hotword_retriever 开关支持 asr_hotword"
```

---

## Task 6: model.py 打点记录每条检索耗时

**Files:**
- Modify: `qwen_asr/joint/model.py`（顶部加 `import time`；transcribe 热词分支打点）

**Interfaces:**
- Produces: 每条 record 新增字段 `hotword_retrieve_ms`（float，毫秒），随 transcribe 返回值进 detail jsonl

- [ ] **Step 1: 顶部加 `import time`**

在 `qwen_asr/joint/model.py:2`（`import json` 上一行或 `import os` 附近）加：

```python
import time
```

具体：将开头三行
```python
import json
import os
import shutil
```
改为
```python
import json
import os
import shutil
import time
```

- [ ] **Step 2: transcribe 热词分支打点**

将 `qwen_asr/joint/model.py:293-300` 的热词分支：

```python
        if need_llm and hotword_retriever is not None:
            contexts = []
            for rec in records:
                src = next((name for name in ("ctc", "rnnt") if rec.get(f"{name}_text")), "")
                words = hotword_retriever.retrieve(rec.get(f"{src}_text", ""), topk=hotword_topk) if src else []
                rec["hotwords"] = words
                rec["hotword_source"] = src
                contexts.append(hotword_prompt(words, base_prompt))
```

替换为：

```python
        if need_llm and hotword_retriever is not None:
            contexts = []
            for rec in records:
                src = next((name for name in ("ctc", "rnnt") if rec.get(f"{name}_text")), "")
                if src:
                    t0 = time.perf_counter()
                    words = hotword_retriever.retrieve(rec.get(f"{src}_text", ""), topk=hotword_topk)
                    rec["hotword_retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                else:
                    words = []
                    rec["hotword_retrieve_ms"] = 0.0
                rec["hotwords"] = words
                rec["hotword_source"] = src
                contexts.append(hotword_prompt(words, base_prompt))
```

- [ ] **Step 3: 验证 import 无误**

Run: `PYTHONPATH=. python3 -c "from qwen_asr.joint.model import Qwen3ASRJointModel; print('import OK')"`
Expected: `import OK`

- [ ] **Step 4: 验证打点逻辑（mock retriever 走 transcribe 热词分支）**

Run: `PYTHONPATH=. python3 -c "
class _MockRet:
    def retrieve(self, q, topk=10): return ['x']
import inspect
from qwen_asr.joint import model
src = inspect.getsource(model.Qwen3ASRJointModel.transcribe)
assert 'hotword_retrieve_ms' in src and 'time.perf_counter' in src, '打点未写入'
print('打点已写入 transcribe')
"`
Expected: `打点已写入 transcribe`

- [ ] **Step 5: Commit**

```bash
git add qwen_asr/joint/model.py
git commit -m "model.py 记录每条音频热词检索耗时到 hotword_retrieve_ms"
```

---

## Task 7: 评测汇总检索耗时 + 透传开关

**Files:**
- Modify: `qwen_asr/tools/hotword_eval.py`（evaluate 收集 ms；write_summary 打印统计）
- Modify: `finetuning/hotword_eval.sh`（声明并透传 `--hotword_retriever`）

**Interfaces:**
- Consumes: detail jsonl 每条 `hotword_retrieve_ms`（Task 6 产出）
- Produces: summary 新增「检索耗时」小节（mean/p50/p95/max/总/样本数）

- [ ] **Step 1: evaluate 收集 `hotword_retrieve_ms`**

在 `qwen_asr/tools/hotword_eval.py` 的 `evaluate(args)` 函数中，`counts = Counts()` 下一行加：

```python
    retrieve_ms: List[float] = []
```

并在循环体内 `counts.samples += 1` 下一行加：

```python
        ms = obj.get("hotword_retrieve_ms")
        if ms is not None:
            retrieve_ms.append(float(ms))
```

并将函数末尾 `return counts, badcases, missing` 改为：

```python
    return counts, badcases, missing, retrieve_ms
```

- [ ] **Step 2: main 适配返回值并传给 write_summary**

将 `qwen_asr/tools/hotword_eval.py` 的 `main()` 中：

```python
    counts, badcases, missing = evaluate(args)
    write_summary(args.output_path, counts, missing)
```

改为：

```python
    counts, badcases, missing, retrieve_ms = evaluate(args)
    write_summary(args.output_path, counts, missing, retrieve_ms)
```

- [ ] **Step 3: write_summary 加检索耗时统计**

在 `qwen_asr/tools/hotword_eval.py` 的 `write_summary(path, counts, missing)` 函数签名改为：

```python
def write_summary(path: str, counts: Counts, missing: int, retrieve_ms: List[float]):
```

并在函数内 `print("误注入识别热词数：...")` 之后、函数结束前加：

```python
    print("", file=f)
    print("检索耗时：", file=f)
    if retrieve_ms:
        sv = sorted(retrieve_ms)
        n = len(sv)
        p50 = sv[min(n - 1, int(n * 0.5))]
        p95 = sv[min(n - 1, int(n * 0.95))]
        print(f"样本数：{n}", file=f)
        print(f"mean：{sum(sv) / n:.3f}ms", file=f)
        print(f"p50：{p50:.3f}ms", file=f)
        print(f"p95：{p95:.3f}ms", file=f)
        print(f"max：{sv[-1]:.3f}ms", file=f)
        print(f"总耗时：{sum(sv):.1f}ms", file=f)
    else:
        print("无检索耗时记录（detail 缺 hotword_retrieve_ms 字段）", file=f)
```

- [ ] **Step 4: 验证耗时统计（造一条假 detail）**

Run: `PYTHONPATH=. python3 -c "
import json, os, tempfile
d = tempfile.mkdtemp()
detail = os.path.join(d, 'detail.jsonl')
with open(detail, 'w', encoding='utf-8') as f:
    f.write(json.dumps({'utt_id':'u1','llm_text':'撒贝宁','hotword_llm_text':'撒贝宁','hotwords':['撒贝宁'],'hotword_retrieve_ms':1.23}, ensure_ascii=False) + '\n')
    f.write(json.dumps({'utt_id':'u2','llm_text':'康辉','hotword_llm_text':'康辉','hotwords':['康辉'],'hotword_retrieve_ms':5.67}, ensure_ascii=False) + '\n')
with open(os.path.join(d,'ref.txt'),'w',encoding='utf-8') as f:
    f.write('u1\t撒贝宁\nu2\t康辉\n')
with open(os.path.join(d,'tgt.txt'),'w',encoding='utf-8') as f:
    f.write('u1\t撒贝宁\nu2\t康辉\n')
import subprocess, sys
r = subprocess.run([sys.executable, 'qwen_asr/tools/hotword_eval.py','--ref_path',os.path.join(d,'ref.txt'),'--detail_path',detail,'--target_hotword_file',os.path.join(d,'tgt.txt'),'--output_path',os.path.join(d,'out.txt'),'--badcase_path',os.path.join(d,'bc.txt')], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
out = open(os.path.join(d,'out.txt'),encoding='utf-8').read()
assert '检索耗时' in out and 'p95' in out and 'mean' in out, out
print('OK 统计写入')
print(out[out.index('检索耗时'):])
"`
Expected: `OK 统计写入` 且输出含 mean/p50/p95/max

- [ ] **Step 5: hotword_eval.sh 声明并透传开关**

在 `finetuning/hotword_eval.sh` 的 `hotword_pinyin_style="normal"` 下一行（第 23 行后）加：

```bash
hotword_retriever="pinyin"
```

在 `arg_map` 数组中 `[--hotword_pinyin_style]=hotword_pinyin_style` 那行后加：

```bash
    [--hotword_retriever]=hotword_retriever
```

在 `infer_cmd` 数组中 `--hotword_pinyin_style "${hotword_pinyin_style}"` 那行后加：

```bash
        --hotword_retriever "${hotword_retriever}"
```

- [ ] **Step 6: 验证 sh 语法**

Run: `bash -n finetuning/hotword_eval.sh && echo "syntax OK"`
Expected: `syntax OK`

- [ ] **Step 7: Commit**

```bash
git add qwen_asr/tools/hotword_eval.py finetuning/hotword_eval.sh
git commit -m "评测汇总每条检索耗时, hotword_eval.sh 透传 retriever 开关"
```

---

## 完成后验证（端到端对比）

在真实数据上分别跑两套 retriever 对比（需要模型 ckpt 与数据集就绪）：

```bash
# pinyin 基线
bash finetuning/hotword_eval.sh --hotword_retriever pinyin --output_dir /cfs/data/private/WangYaoChi/test_out/joint_ctc_14_hotword_4/aishell_pinyin

# asr_hotword 复现
bash finetuning/hotword_eval.sh --hotword_retriever asr_hotword --output_dir /cfs/data/private/WangYaoChi/test_out/joint_ctc_14_hotword_4/aishell_asrhotword
```

对比两份 `<output_dir>/hotword_eval.txt` 的 R@1/3/5/10、召回准确率、误召回、识别率、检索耗时。

---

## Self-Review 记录

**1. Spec 覆盖：**
- 子包保真拷贝 4 文件 → Task 1-3 ✓
- adapter AsrHotwordRetriever（retrieve 接口、from_file、阈值默认 0.65）→ Task 4 ✓
- infer.py 开关切换两套 retriever → Task 5 ✓
- model.py 每条检索耗时 → Task 6 ✓
- hotword_eval.py 耗时统计 → Task 7 ✓
- hotword_eval.sh 透传 → Task 7 ✓
- 算法常量保真（SIMILAR_PHONEMES / DP 0.5 / 0.55 / 0.65）→ calc.py & retriever.py 默认值 ✓
- 召回阈值策略（≥0.65 才召回，不足 topk 返回全部）→ retriever.py ✓

**2. 占位符扫描：** 无 TBD/TODO，每个代码步骤含完整代码，验证命令含期望输出。✓

**3. 类型一致性：** `FastRAG.search` 返回 `List[(hw, score, approx_end)]` 与 retriever 解包 `for hw, _score, approx_end in fast_results` 一致；`fuzzy_substring_search_constrained` 返回 `[(score, s, e)]` 与 retriever 解包 `for score, _s, _e` 一致；`retrieve(query, topk)->List[str]` 与 model.py 调用 `retrieve(..., topk=hotword_topk)` 一致；`evaluate` 返回四元组与 `main` 解包一致。✓
