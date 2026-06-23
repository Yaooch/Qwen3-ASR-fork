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
