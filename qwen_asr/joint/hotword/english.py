# coding: utf-8
"""英文音素补充召回：音素粗筛 + 单词边界精筛。"""
import re
import sqlite3
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import rapidfuzz.fuzz as _fuzz
import rapidfuzz.process as _process
from rapidfuzz.distance import Levenshtein as _Levenshtein


_WORD_RE = re.compile(r"[a-z]+")
_PHONE_GROUPS = [
    {"i", "ɪ", "e", "ɛ", "æ"},
    {"ə", "ɚ", "ɝ", "ʌ"},
    {"ɑ", "ɔ"},
    {"o", "oʊ", "ʊ", "u"},
    {"p", "b"},
    {"t", "d"},
    {"k", "g"},
    {"f", "v"},
    {"θ", "ð"},
    {"s", "z"},
    {"ʃ", "ʒ"},
    {"tʃ", "dʒ"},
    {"m", "n", "ŋ"},
]
_PHONE_CLASS = {
    phone: f"C{idx}"
    for idx, group in enumerate(_PHONE_GROUPS)
    for phone in group
}


def _strip_stress(phone: str) -> str:
    return phone.lstrip("ˈˌ")


@lru_cache(maxsize=1)
def _resources():
    from gruut.lang import find_lang_dir, get_settings

    lang_dir = find_lang_dir("en-us")
    if lang_dir is None:
        raise RuntimeError("gruut 缺少 en-us 发音词典")
    db = sqlite3.connect(str(lang_dir / "lexicon.db"))
    lexicon = {}
    try:
        rows = db.execute(
            "SELECT word, phonemes FROM word_phonemes "
            "ORDER BY word, (role != ''), pron_order"
        )
        for word, phonemes in rows:
            lexicon.setdefault(word, tuple(_strip_stress(p) for p in phonemes.split()))
    finally:
        db.close()

    guess = get_settings("en-us").guess_phonemes
    if guess is None:
        raise RuntimeError("gruut 缺少 en-us G2P 模型")
    return lexicon, guess


def _is_english(text: str) -> bool:
    return bool(_WORD_RE.search(text.lower())) and not any("一" <= c <= "鿿" for c in text)


def _map_phones(phones: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(_PHONE_CLASS.get(phone, phone) for phone in phones)


def _threshold(phone_count: int) -> float:
    if phone_count <= 3:
        return 0.95
    if phone_count <= 5:
        return 0.85
    if phone_count <= 7:
        return 0.75
    return 0.65


class EnglishPhoneMatcher:
    """只返回一个达到长度阈值的高置信英文音素候选。"""

    def __init__(self, hotwords: List[str]):
        self._phones: Dict[str, Tuple[str, ...]] = {}
        self._classes: Dict[str, Tuple[str, ...]] = {}
        self._word_counts: Dict[str, int] = {}
        self._word_cache: Dict[str, Tuple[str, ...]] = {}
        english_words = [word for word in hotwords if _is_english(word)]
        if not english_words:
            self._lexicon = {}
            self._guess = None
            self._max_words = 0
            return

        self._lexicon, self._guess = _resources()
        self._max_words = 0
        for word in english_words:
            phone_words = self._words(word)
            phones = tuple(phone for item in phone_words for phone in item)
            if phones:
                self._phones[word] = phones
                self._classes[word] = _map_phones(phones)
                self._word_counts[word] = len(phone_words)
                self._max_words = max(self._max_words, len(phone_words))

    def _word_phones(self, word: str) -> Tuple[str, ...]:
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached
        phones = self._lexicon.get(word)
        if phones is None:
            phones = tuple(_strip_stress(p) for p in (self._guess(word) or ()))
        phones = tuple(phone for phone in phones if phone)
        self._word_cache[word] = phones
        return phones

    def _words(self, text: str) -> List[Tuple[str, ...]]:
        text = unicodedata.normalize("NFKC", text or "").lower()
        return [
            phones
            for word in _WORD_RE.findall(text)
            if (phones := self._word_phones(word))
        ]

    def retrieve(self, query: str) -> Optional[str]:
        if not self._phones:
            return None
        query_words = self._words(query)
        query_phones = tuple(phone for item in query_words for phone in item)
        if not query_phones:
            return None

        spans = []
        for start in range(len(query_words)):
            phones = ()
            for end in range(start, min(len(query_words), start + self._max_words + 2)):
                phones += query_words[end]
                spans.append((end - start + 1, phones, _map_phones(phones)))

        winner = None
        winner_score = 0.0
        coarse = _process.extract(
            query_phones,
            self._phones,
            scorer=_fuzz.partial_ratio,
            score_cutoff=35,
            limit=20,
        )
        for _value, _score, word in coarse:
            target = self._phones[word]
            target_class = self._classes[word]
            target_words = self._word_counts[word]
            best = 0.0
            for span_words, span, span_class in spans:
                if abs(span_words - target_words) > 2:
                    continue
                size = max(len(target), len(span))
                exact = 1.0 - _Levenshtein.distance(target, span) / size
                broad = 1.0 - _Levenshtein.distance(target_class, span_class) / size
                best = max(best, (exact + broad) / 2.0)
            if best >= _threshold(len(target)) and best > winner_score:
                winner = word
                winner_score = best
        return winner
