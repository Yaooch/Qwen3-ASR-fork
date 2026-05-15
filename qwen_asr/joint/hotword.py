# qwen_asr/joint/hotword.pinyin
"""热词召回：用粗识别文本做拼音滑窗检索。"""
from dataclasses import dataclass
from typing import List, Optional, Sequence


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
    pinyin: List[str]
    tone_pinyin: List[str]
    initials: List[str]
    finals: List[str]


class HotwordRetriever:
    """拼音优先的实体热词召回。"""

    def __init__(
        self,
        hotwords: List[str],
        pinyin_style: str = "normal",
        min_score: Optional[float] = None,
    ):
        self.hotwords = [h.strip() for h in hotwords if h.strip()]
        self.pinyin_style = pinyin_style
        self.min_score = min_score
        self._entries = []
        self._lazy_pinyin = None
        self._style = None
        self._tone3_style = None
        self._initials_style = None
        self._finals_style = None
        self._pinyin_index = {}
        self._initial_index = {}
        self._final_index = {}

        try:
            from pypinyin import Style, lazy_pinyin
            from rapidfuzz import fuzz
        except ImportError as exc:
            raise RuntimeError("缺少依赖 pypinyin 或 rapidfuzz，请先安装后再使用热词召回。") from exc

        self._fuzz = fuzz
        self._lazy_pinyin = lazy_pinyin
        self._style = self._parse_style(Style, pinyin_style)
        self._tone3_style = Style.TONE3
        self._initials_style = Style.INITIALS
        self._finals_style = Style.FINALS
        self._entries = [self._entry(w) for w in self.hotwords]
        self._build_index()

    @classmethod
    def from_file(cls, path: str, **kwargs):
        with open(path, "r", encoding="utf-8") as f:
            hotwords = [line.strip() for line in f if line.strip()]
        return cls(hotwords, **kwargs)

    def retrieve(self, query: str, topk: int = 10) -> List[str]:
        if not self.hotwords or not query:
            return []
        return self._retrieve_pinyin(query, topk=topk)

    def _retrieve_pinyin(self, query: str, topk: int) -> List[str]:
        q_py = self._tokens(query, self._style)
        if not q_py:
            return []

        q_tone = self._tokens(query, self._tone3_style)
        q_initials = self._tokens(query, self._initials_style)
        q_finals = self._tokens(query, self._finals_style)

        best = {}
        for entry in self._candidates(q_py, q_initials, q_finals, topk):
            if not entry.pinyin:
                continue
            score = self._best_score(entry, q_py, q_tone, q_initials, q_finals)
            threshold = self.min_score if self.min_score is not None else self._threshold(len(entry.pinyin))
            if score >= threshold and score > best.get(entry.word, 0.0):
                best[entry.word] = score

        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in ranked[:topk]]

    def _candidates(
        self,
        q_py: Sequence[str],
        q_initials: Sequence[str],
        q_finals: Sequence[str],
        topk: int,
    ) -> List[HotwordEntry]:
        """先做拼音粗排，避免每条语音都全表滑窗精算。"""
        if len(self._entries) <= 1024:
            return self._entries

        q_len = len(q_py)
        limit = max(64, topk * 16)
        q_text = " ".join(q_py)
        scored = []
        for entry in self._indexed_candidates(q_py, q_initials, q_finals, q_len, limit * 4):
            score = self._fuzz.partial_ratio(q_text, " ".join(entry.pinyin))
            if score >= 35:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _score, entry in scored[:limit]]

    def _indexed_candidates(
        self,
        q_py: Sequence[str],
        q_initials: Sequence[str],
        q_finals: Sequence[str],
        q_len: int,
        limit: int,
    ) -> List[HotwordEntry]:
        scores = {}
        entries = {}

        def add(cands, weight: int) -> None:
            for entry in cands:
                if len(entry.pinyin) > q_len + 1:
                    continue
                scores[entry.word] = scores.get(entry.word, 0) + weight
                entries[entry.word] = entry

        for pinyin in set(q_py):
            add(self._pinyin_index.get(pinyin, []), 4)
        for initial in set(q_initials):
            if not initial:
                continue
            add(self._initial_index.get(initial, []), 1)
            for near in NEAR_INITIALS.get(initial, set()):
                add(self._initial_index.get(near, []), 1)
        for final in set(q_finals):
            if not final:
                continue
            add(self._final_index.get(final, []), 1)
            for near in NEAR_FINALS.get(final, set()):
                add(self._final_index.get(near, []), 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [entries[word] for word, _score in ranked[:limit]]

    def _best_score(
        self,
        entry: HotwordEntry,
        q_py: List[str],
        q_tone: List[str],
        q_initials: List[str],
        q_finals: List[str],
    ) -> float:
        h_len = len(entry.pinyin)
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
        pinyin: List[str],
        tone_pinyin: List[str],
        initials: List[str],
        finals: List[str],
    ) -> float:
        py_score = self._ratio(entry.pinyin, pinyin)
        tone_score = self._ratio(entry.tone_pinyin, tone_pinyin)
        initial_score = self._ratio(entry.initials, initials)
        final_score = self._ratio(entry.finals, finals)
        aligned_score = self._aligned_score(entry, pinyin, initials, finals)
        length_penalty = abs(len(entry.pinyin) - len(pinyin)) * 0.08

        score = (
            0.45 * py_score
            + 0.15 * tone_score
            + 0.15 * initial_score
            + 0.10 * final_score
            + 0.15 * aligned_score
            - length_penalty
        )
        if entry.pinyin == pinyin:
            score += 0.06
        if entry.tone_pinyin == tone_pinyin:
            score += 0.03
        score = max(score, self._short_name_score(entry, pinyin, initials, finals) - length_penalty)
        return min(1.0, max(0.0, score))

    def _aligned_score(
        self,
        entry: HotwordEntry,
        pinyin: List[str],
        initials: List[str],
        finals: List[str],
    ) -> float:
        if len(entry.pinyin) != len(pinyin) or not pinyin:
            return 0.0
        scores = []
        for h_py, q_py, h_i, q_i, h_f, q_f in zip(
            entry.pinyin, pinyin, entry.initials, initials, entry.finals, finals
        ):
            scores.append(self._syllable_score(h_py, q_py, h_i, q_i, h_f, q_f))
        return sum(scores) / len(scores)

    def _short_name_score(
        self,
        entry: HotwordEntry,
        pinyin: List[str],
        initials: List[str],
        finals: List[str],
    ) -> float:
        h_len = len(entry.pinyin)
        q_len = len(pinyin)
        if h_len not in (3, 4) or q_len < h_len - 1 or q_len >= h_len:
            return 0.0

        # 人名常见错误是姓或中间一字被吞掉，保留名的音节仍然很准。
        candidates = []
        for start in range(0, h_len - q_len + 1):
            end = start + q_len
            sub = HotwordEntry(
                word=entry.word,
                pinyin=entry.pinyin[start:end],
                tone_pinyin=entry.tone_pinyin[start:end],
                initials=entry.initials[start:end],
                finals=entry.finals[start:end],
            )
            candidates.append(self._aligned_score(sub, pinyin, initials, finals))
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
        return self._fuzz.ratio(h_py, q_py) / 100.0 * 0.65

    def _entry(self, word: str) -> HotwordEntry:
        return HotwordEntry(
            word=word,
            pinyin=self._tokens(word, self._style),
            tone_pinyin=self._tokens(word, self._tone3_style),
            initials=self._tokens(word, self._initials_style),
            finals=self._tokens(word, self._finals_style),
        )

    def _build_index(self) -> None:
        for entry in self._entries:
            for pinyin in set(entry.pinyin):
                self._pinyin_index.setdefault(pinyin, []).append(entry)
            for initial in {x for x in entry.initials if x}:
                self._initial_index.setdefault(initial, []).append(entry)
            for final in {x for x in entry.finals if x}:
                self._final_index.setdefault(final, []).append(entry)

    def _tokens(self, text: str, style) -> List[str]:
        tokens = []
        buf = []
        for ch in text.strip():
            if ch.isspace():
                self._flush(buf, tokens)
            elif "\u4e00" <= ch <= "\u9fff":
                self._flush(buf, tokens)
                items = self._lazy_pinyin(ch, style=style, errors="ignore", strict=False)
                if items:
                    tokens.append(items[0].lower() or "_")
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
        return self._fuzz.ratio(left_text, right_text) / 100.0

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
