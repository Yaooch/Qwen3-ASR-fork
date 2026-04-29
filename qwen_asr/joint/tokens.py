# qwen_asr/joint/tokens.py
"""CTC 词表/SentencePiece 相关的共享工具：训练 collator 和推理脚本都用。"""
import json
import os
import re
from typing import Dict, List, Optional

CJK_PATTERN = re.compile(r'([\u4e00-\u9fff])')


def load_bpe_vocab(vocab_path: str) -> Dict[str, int]:
    """BPE vocab 文件格式：每行 'token id'"""
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                vocab[parts[0]] = int(parts[1])
    return vocab


def load_sp_model(sp_model_path: str):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    return sp


def text_to_ctc_ids(text: str, vocab: Dict[str, int], sp_model) -> List[int]:
    """模仿 WeNet 的 __tokenize_by_bpe_model：CJK 逐字，其余走 SentencePiece。"""
    text = text.strip()
    if not text:
        return []
    ids: List[int] = []
    chunks = [w for w in CJK_PATTERN.split(text) if len(w.strip()) > 0]
    unk_id = vocab.get("<unk>", 1)
    for ch in chunks:
        if CJK_PATTERN.fullmatch(ch):
            ids.append(vocab.get(ch, unk_id))
        else:
            pieces = sp_model.encode_as_pieces(ch.upper())
            for p in pieces:
                if p in vocab:
                    ids.append(vocab[p])
                else:
                    clean = p.replace("▁", "")
                    ids.append(vocab.get(clean, unk_id))
    return ids


def ids_to_text(ids: List[int], id_to_token: Dict[int, str]) -> str:
    tokens = [id_to_token.get(i, "") for i in ids]
    return "".join(tokens).replace("▁", " ").strip().lower()


def build_id_to_token(vocab: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in vocab.items()}
