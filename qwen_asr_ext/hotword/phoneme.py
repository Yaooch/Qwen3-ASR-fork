# coding: utf-8
"""文本音素转换与边界约束 DP 精筛。"""
import unicodedata
from dataclasses import dataclass
from typing import List, Literal, Tuple

from pypinyin import Style, pinyin


@dataclass(frozen=True)
class Phoneme:
    value: str
    lang: Literal["zh", "en", "num", "other"]
    is_word_start: bool = False
    is_word_end: bool = False

    @property
    def is_tone(self) -> bool:
        return self.value.isdigit()


def get_phoneme_info(text: str) -> List[Phoneme]:
    phonemes: List[Phoneme] = []
    pos = 0
    while pos < len(text):
        char = text[pos]
        if "一" <= char <= "鿿":
            pos = _process_zh(text, pos, phonemes)
        elif "a" <= char.lower() <= "z" or "0" <= char <= "9":
            pos = _process_en_num(text, pos, phonemes)
        else:
            if unicodedata.category(char).startswith("L"):
                phonemes.append(Phoneme(char.lower(), "other", True, True))
            pos += 1
    return phonemes


def _process_zh(text: str, pos: int, phonemes: List[Phoneme]) -> int:
    end = pos + 1
    while end < len(text) and "一" <= text[end] <= "鿿":
        end += 1
    fragment = text[pos:end]
    try:
        initials = pinyin(fragment, style=Style.INITIALS, strict=False, errors="ignore")
        finals = pinyin(fragment, style=Style.FINALS, strict=False, errors="ignore")
        tones = pinyin(
            fragment,
            style=Style.TONE3,
            neutral_tone_with_five=True,
            errors="ignore",
        )
        for index in range(min(len(fragment), len(initials), len(finals), len(tones))):
            initial, final, tone = initials[index][0], finals[index][0], tones[index][0]
            items = []
            if initial:
                items.append(Phoneme(initial, "zh", is_word_start=True))
            if final:
                items.append(Phoneme(final, "zh", is_word_start=not initial))
            if tone and tone[-1].isdigit():
                items.append(Phoneme(tone[-1], "zh", is_word_end=True))
            if not items:
                items.append(Phoneme(fragment[index], "zh", True, True))
            phonemes.extend(items)
    except Exception:
        phonemes.extend(Phoneme(char, "zh", True, True) for char in fragment)
    return end


def _process_en_num(text: str, pos: int, phonemes: List[Phoneme]) -> int:
    start = pos
    while pos < len(text):
        char = text[pos]
        if not ("a" <= char.lower() <= "z" or "0" <= char <= "9"):
            break
        if pos > start:
            previous = text[pos - 1]
            if (
                (previous.islower() and char.isupper())
                or (previous.isalpha() and char.isdigit())
                or (previous.isdigit() and char.isalpha())
            ):
                break
        pos += 1
    token = text[start:pos].lower()
    lang = "num" if token.isdigit() else "en"
    for index, char in enumerate(token):
        phonemes.append(
            Phoneme(
                char,
                lang,
                is_word_start=index == 0,
                is_word_end=index == len(token) - 1,
            )
        )
    return pos

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


def fuzzy_substring_search_constrained(hw_phonemes: List[Phoneme], input_phonemes: List[Phoneme],
                                        threshold: float = 0.6) -> List[Tuple[float, int, int]]:
    n = len(hw_phonemes)
    m = len(input_phonemes)
    if n == 0 or m == 0:
        return []

    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    path = [[(0, 0)] * (m + 1) for _ in range(n + 1)]

    input_vals = [p.value for p in input_phonemes]
    input_langs = [p.lang for p in input_phonemes]
    input_starts = [p.is_word_start for p in input_phonemes]
    hw_vals = [p.value for p in hw_phonemes]
    hw_langs = [p.lang for p in hw_phonemes]
    hw_phones = [p.is_tone for p in hw_phonemes]

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
        if not input_phonemes[j - 1].is_word_end:
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
