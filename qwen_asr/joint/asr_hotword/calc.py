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
