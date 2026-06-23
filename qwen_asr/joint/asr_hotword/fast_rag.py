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
