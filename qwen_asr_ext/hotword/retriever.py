# coding: utf-8
"""音素热词的 FastRAG 粗筛与两阶段检索入口。"""
from typing import Dict, List, Tuple

import rapidfuzz.fuzz as _fuzz
import rapidfuzz.distance.OSA as _OSA
import rapidfuzz.process as _process

from .english import EnglishPhoneMatcher
from .phoneme import Phoneme, fuzzy_substring_search_constrained, get_phoneme_info


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

        best: Dict[str, float] = {}
        for hw, positions in seen.items():
            for approx_end in positions:
                for hw_phonemes in self._phonemes.get(hw, []):
                    window_size = len(hw_phonemes) + 10
                    win_start = max(0, approx_end - window_size)
                    win_end = min(len(input_phonemes), approx_end + 5)
                    local_input = input_phonemes[win_start:win_end]
                    for score, _s, _e in fuzzy_substring_search_constrained(
                        hw_phonemes, local_input, threshold=self.fast_threshold
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
