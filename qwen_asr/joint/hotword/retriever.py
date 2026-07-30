# coding: utf-8
"""热词检索入口：FastRAG 粗筛 + 边界约束 DP 精筛。"""
from typing import Dict, List

from .phoneme import Phoneme, get_phoneme_info
from .calc import fuzzy_substring_search_constrained
from .english import EnglishPhoneMatcher
from .fast_rag import FastRAG


class HotwordRetriever:
    """音素级两层热词检索。"""

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
        self._english = EnglishPhoneMatcher(self.hotwords)

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "HotwordRetriever":
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
        words = [w for w, _ in ranked[:topk]]
        if len(words) < topk:
            phone_word = self._english.retrieve(query)
            if phone_word and phone_word not in words:
                words.append(phone_word)
        return words
