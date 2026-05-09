# qwen_asr/joint/hotword.py
"""热词召回：用粗识别文本做拼音滑窗检索。"""
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional


NEAR_INITIALS = {
    "b": {"p"},
    "p": {"b"},
    "d": {"t"},
    "t": {"d"},
    "g": {"k"},
    "k": {"g"},
    "z": {"zh", "c"},
    "zh": {"z", "ch"},
    "c": {"ch", "z"},
    "ch": {"c", "zh"},
    "s": {"sh"},
    "sh": {"s"},
    "l": {"n", "r"},
    "n": {"l"},
    "r": {"l"},
    "f": {"h"},
    "h": {"f"},
}

NEAR_FINALS = {
    "en": {"eng"},
    "eng": {"en"},
    "in": {"ing"},
    "ing": {"in"},
    "an": {"ang"},
    "ang": {"an"},
    "uan": {"uang"},
    "uang": {"uan"},
    "ian": {"iang"},
    "iang": {"ian"},
    "ong": {"eng"},
    "eng": {"en", "ong"},
}


@dataclass
class HotwordEntry:
    word: str
    py: List[str]
    tone_py: List[str]
    initials: List[str]
    finals: List[str]


class HotwordRetriever:
    """拼音优先的实体热词召回。"""

    def __init__(
        self,
        hotwords: List[str],
        scorer: str = "pinyin",
        pinyin_style: str = "normal",
        min_score: Optional[float] = None,
    ):
        self.hotwords = [h.strip() for h in hotwords if h.strip()]
        self.scorer = scorer
        self.pinyin_style = pinyin_style
        self.min_score = min_score
        self._entries = []
        self._lazy_pinyin = None
        self._style = None
        self._tone3_style = None
        self._initials_style = None
        self._finals_style = None

        try:
            from rapidfuzz import fuzz
            self._fuzz = fuzz
        except ImportError:
            self._fuzz = None

        if scorer == "pinyin":
            try:
                from pypinyin import Style, lazy_pinyin
                self._lazy_pinyin = lazy_pinyin
                self._style = self._parse_style(Style, pinyin_style)
                self._tone3_style = Style.TONE3
                self._initials_style = Style.INITIALS
                self._finals_style = Style.FINALS
                self._entries = [self._entry(w) for w in self.hotwords]
            except ImportError:
                print("未安装 pypinyin，热词检索改用字符相似度")
                self.scorer = "fuzz"

        if self.scorer == "edit":
            import editdistance
            self._ed = editdistance

    @classmethod
    def from_file(cls, path: str, **kwargs):
        with open(path, "r", encoding="utf-8") as f:
            hotwords = [line.strip() for line in f if line.strip()]
        return cls(hotwords, **kwargs)

    def retrieve(self, query: str, topk: int = 10) -> List[str]:
        if not self.hotwords or not query:
            return []
        if self.scorer == "pinyin":
            return self._retrieve_pinyin(query, topk=topk)
        if self.scorer == "edit":
            return self._retrieve_edit(query, topk=topk)
        return self._retrieve_fuzz(query, topk=topk)

    def _retrieve_pinyin(self, query: str, topk: int) -> List[str]:
        q_py = self._tokens(query, self._style)
        if not q_py:
            return []

        q_tone = self._tokens(query, self._tone3_style)
        q_initials = self._tokens(query, self._initials_style)
        q_finals = self._tokens(query, self._finals_style)

        best = {}
        for entry in self._entries:
            if not entry.py:
                continue
            score = self._best_score(entry, q_py, q_tone, q_initials, q_finals)
            threshold = self.min_score if self.min_score is not None else self._threshold(len(entry.py))
            if score >= threshold and score > best.get(entry.word, 0.0):
                best[entry.word] = score

        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in ranked[:topk]]

    def _best_score(
        self,
        entry: HotwordEntry,
        q_py: List[str],
        q_tone: List[str],
        q_initials: List[str],
        q_finals: List[str],
    ) -> float:
        h_len = len(entry.py)
        q_len = len(q_py)
        min_len = max(1, h_len - 1)
        max_len = min(q_len, h_len + 2)
        if h_len <= 2:
            min_len = h_len
            max_len = min(q_len, h_len + 1)

        best = 0.0
        for win_len in range(min_len, max_len + 1):
            for start in range(0, q_len - win_len + 1):
                end = start + win_len
                score = self._window_score(
                    entry,
                    q_py[start:end],
                    q_tone[start:end],
                    q_initials[start:end],
                    q_finals[start:end],
                )
                if score > best:
                    best = score
        return best

    def _window_score(
        self,
        entry: HotwordEntry,
        py: List[str],
        tone_py: List[str],
        initials: List[str],
        finals: List[str],
    ) -> float:
        py_score = self._ratio(entry.py, py)
        tone_score = self._ratio(entry.tone_py, tone_py)
        initial_score = self._ratio(entry.initials, initials)
        final_score = self._ratio(entry.finals, finals)
        aligned_score = self._aligned_score(entry, py, initials, finals)
        length_penalty = abs(len(entry.py) - len(py)) * 0.08

        score = (
            0.45 * py_score
            + 0.15 * tone_score
            + 0.15 * initial_score
            + 0.10 * final_score
            + 0.15 * aligned_score
            - length_penalty
        )
        if entry.py == py:
            score += 0.06
        if entry.tone_py == tone_py:
            score += 0.03
        score = max(score, self._short_name_score(entry, py, initials, finals) - length_penalty)
        return min(1.0, max(0.0, score))

    def _aligned_score(
        self,
        entry: HotwordEntry,
        py: List[str],
        initials: List[str],
        finals: List[str],
    ) -> float:
        if len(entry.py) != len(py) or not py:
            return 0.0
        scores = []
        for h_py, q_py, h_i, q_i, h_f, q_f in zip(
            entry.py, py, entry.initials, initials, entry.finals, finals
        ):
            scores.append(self._syllable_score(h_py, q_py, h_i, q_i, h_f, q_f))
        return sum(scores) / len(scores)

    def _short_name_score(
        self,
        entry: HotwordEntry,
        py: List[str],
        initials: List[str],
        finals: List[str],
    ) -> float:
        h_len = len(entry.py)
        q_len = len(py)
        if h_len not in (3, 4) or q_len < h_len - 1 or q_len >= h_len:
            return 0.0

        # 人名常见错误是姓或中间一字被吞掉，保留名的音节仍然很准。
        candidates = []
        for start in range(0, h_len - q_len + 1):
            end = start + q_len
            sub = HotwordEntry(
                word=entry.word,
                py=entry.py[start:end],
                tone_py=entry.tone_py[start:end],
                initials=entry.initials[start:end],
                finals=entry.finals[start:end],
            )
            candidates.append(self._aligned_score(sub, py, initials, finals))
        best = max(candidates) if candidates else 0.0
        if h_len == 3 and q_len == 2:
            return best * 0.92
        return best * 0.86

    def _syllable_score(
        self,
        h_py: str,
        q_py: str,
        h_initial: str,
        q_initial: str,
        h_final: str,
        q_final: str,
    ) -> float:
        if h_py == q_py:
            return 1.0
        initial_same = h_initial == q_initial
        final_same = h_final == q_final
        initial_near = q_initial in NEAR_INITIALS.get(h_initial, set())
        final_near = q_final in NEAR_FINALS.get(h_final, set())

        if initial_same and final_near:
            return 0.88
        if initial_near and final_same:
            return 0.86
        if initial_near and final_near:
            return 0.78
        if final_same:
            return 0.72
        if initial_same:
            return 0.62
        return SequenceMatcher(None, h_py, q_py).ratio() * 0.65

    def _entry(self, word: str) -> HotwordEntry:
        return HotwordEntry(
            word=word,
            py=self._tokens(word, self._style),
            tone_py=self._tokens(word, self._tone3_style),
            initials=self._tokens(word, self._initials_style),
            finals=self._tokens(word, self._finals_style),
        )

    def _tokens(self, text: str, style) -> List[str]:
        tokens = []
        buf = []
        for ch in text.strip():
            if ch.isspace():
                self._flush(buf, tokens)
            elif "\u4e00" <= ch <= "\u9fff":
                self._flush(buf, tokens)
                py = self._lazy_pinyin(ch, style=style, errors="ignore", strict=False)
                if py:
                    tokens.append(py[0].lower() or "_")
            elif ch.isalnum():
                buf.append(ch.lower())
            else:
                self._flush(buf, tokens)
        self._flush(buf, tokens)
        return tokens

    def _flush(self, buf: List[str], tokens: List[str]) -> None:
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    def _ratio(self, left: List[str], right: List[str]) -> float:
        if not left or not right:
            return 0.0
        left_text = " ".join(left)
        right_text = " ".join(right)
        if self._fuzz is not None:
            return self._fuzz.ratio(left_text, right_text) / 100.0
        return SequenceMatcher(None, left_text, right_text).ratio()

    def _threshold(self, length: int) -> float:
        if length <= 1:
            return 0.99
        if length <= 2:
            return 0.86
        if length <= 4:
            return 0.80
        if length <= 8:
            return 0.76
        return 0.70

    def _parse_style(self, style_cls, style_name: str):
        if style_name == "normal":
            return style_cls.NORMAL
        if style_name == "tone3":
            return style_cls.TONE3
        raise ValueError(f"不支持的 pinyin_style: {style_name}")

    def _retrieve_fuzz(self, query: str, topk: int) -> List[str]:
        if self._fuzz is None:
            return self._retrieve_edit(query, topk=topk)
        scored = [(w, self._fuzz.partial_ratio(query, w)) for w in self.hotwords]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [w for w, score in scored[:topk] if score > 60]

    def _retrieve_edit(self, query: str, topk: int) -> List[str]:
        if not hasattr(self, "_ed"):
            try:
                import editdistance
                self._ed = editdistance
            except ImportError:
                scored = [(w, SequenceMatcher(None, query, w).ratio()) for w in self.hotwords]
                scored.sort(key=lambda x: x[1], reverse=True)
                return [w for w, score in scored[:topk] if score > 0.6]

        scored = []
        for w in self.hotwords:
            dist = self._ed.eval(query, w)
            norm = max(len(query), len(w), 1)
            scored.append((w, 1.0 - dist / norm))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [w for w, score in scored[:topk] if score > 0.6]
