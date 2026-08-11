# coding: utf-8
"""文本到带词边界的音素序列。"""
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
