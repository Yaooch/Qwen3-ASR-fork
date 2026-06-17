# qwen_asr/joint/hotword_scorer.py
"""Encoder 热词打分器：热词 token cross-attention 到流式 hs。"""

import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


_HOTWORD_RE = re.compile(r"专属名词\s*[:：]\s*[\[【](.*?)[\]】]")
_PUNCT_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")
def extract_ref(text: str) -> str:
    return (text or "").split("<asr_text>")[-1].strip()


def extract_hotwords(prompt: str) -> List[str]:
    match = _HOTWORD_RE.search(prompt or "")
    if not match:
        return []
    words = []
    for item in re.split(r"[，,、;；]", match.group(1)):
        word = item.strip()
        if word and word not in words:
            words.append(word)
    return words


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def compact_text(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def hotword_label(ref: str, hotword: str) -> int:
    ref_norm = normalize_text(ref)
    word_norm = normalize_text(hotword)
    if not ref_norm or not word_norm:
        return 0
    if word_norm in ref_norm:
        return 1
    # 英文缩写、连字符和空格写法经常不一致，压紧后再匹配一次。
    ref_compact = compact_text(ref)
    word_compact = compact_text(hotword)
    return int(bool(word_compact and word_compact in ref_compact))


def batch_tokenize_hotwords(
    tokenizer,
    hotwords_batch: Sequence[Sequence[str]],
    max_len: int,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 batch 内不等长热词列表转成 [B,K,L] ids/mask/valid。"""
    bsz = len(hotwords_batch)
    max_k = max((len(x) for x in hotwords_batch), default=0)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    ids = torch.full((bsz, max_k, max_len), int(pad_id), dtype=torch.long, device=device)
    token_mask = torch.zeros((bsz, max_k, max_len), dtype=torch.bool, device=device)
    valid = torch.zeros((bsz, max_k), dtype=torch.bool, device=device)
    flat_words, owners = [], []
    for b, words in enumerate(hotwords_batch):
        for k, word in enumerate(words):
            flat_words.append(word)
            owners.append((b, k))
    if not flat_words:
        return ids, token_mask, valid

    tok = tokenizer(
        flat_words,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    flat_ids = tok["input_ids"].to(device=device)
    flat_mask = tok["attention_mask"].to(device=device).bool()
    length = min(max_len, flat_ids.shape[1])
    for row, (b, k) in enumerate(owners):
        ids[b, k, :length] = flat_ids[row, :length]
        token_mask[b, k, :length] = flat_mask[row, :length]
        valid[b, k] = bool(flat_mask[row, :length].any().item())
    return ids, token_mask, valid


def read_hotwords(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class HotwordScorer(nn.Module):
    """轻量 cross-attention 热词分类器。

    输入冻结的热词 token embedding 和流式 encoder hs，输出每个热词的命中 logit。
    """

    def __init__(
        self,
        encoder_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        max_hotword_len: int = 24,
    ):
        super().__init__()
        self.encoder_dim = int(encoder_dim)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.ffn_mult = int(ffn_mult)
        self.dropout_p = float(dropout)
        self.max_hotword_len = int(max_hotword_len)

        self.audio_adapter = nn.Linear(self.encoder_dim, self.embed_dim)
        self.pos = nn.Parameter(torch.zeros(self.max_hotword_len, self.embed_dim))
        self.cross_attn = nn.MultiheadAttention(
            self.embed_dim,
            self.num_heads,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * self.ffn_mult),
            nn.SiLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.embed_dim * self.ffn_mult, self.embed_dim),
        )
        self.ffn_norm = nn.LayerNorm(self.embed_dim)
        self.drop = nn.Dropout(self.dropout_p)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.embed_dim, 1),
        )

    def config(self) -> Dict:
        return {
            "encoder_dim": self.encoder_dim,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_mult": self.ffn_mult,
            "dropout": self.dropout_p,
            "max_hotword_len": self.max_hotword_len,
        }

    def forward(
        self,
        hs: torch.Tensor,
        lens: torch.Tensor,
        hotword_embeds: torch.Tensor,
        hotword_token_mask: torch.Tensor,
        hotword_valid_mask: torch.Tensor = None,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """返回 [B,K] logits。"""
        if hotword_embeds.dim() != 4:
            raise ValueError("hotword_embeds 需要是 [B,K,L,H]")
        bsz, n_words, n_tokens, dim = hotword_embeds.shape
        if dim != self.embed_dim:
            raise ValueError(f"热词 embedding 维度不匹配：{dim} != {self.embed_dim}")
        if n_tokens > self.max_hotword_len:
            raise ValueError(f"热词 token 长度超过上限：{n_tokens} > {self.max_hotword_len}")
        if n_words == 0:
            return hs.new_zeros((bsz, 0), dtype=torch.float32)

        audio = self.audio_adapter(hs.float())
        t = audio.shape[1]
        audio_pad = torch.arange(t, device=audio.device).unsqueeze(0) >= lens.to(audio.device).unsqueeze(1)
        logits = []
        step = max(1, int(chunk_size))
        for start in range(0, n_words, step):
            end = min(n_words, start + step)
            cur_k = end - start
            q = hotword_embeds[:, start:end].float()
            q = q + self.pos[:n_tokens].view(1, 1, n_tokens, dim)
            q = q.reshape(bsz * cur_k, n_tokens, dim)
            key = audio.unsqueeze(1).expand(bsz, cur_k, t, dim).reshape(bsz * cur_k, t, dim)
            key_pad = audio_pad.unsqueeze(1).expand(bsz, cur_k, t).reshape(bsz * cur_k, t)

            attn, _ = self.cross_attn(q, key, key, key_padding_mask=key_pad, need_weights=False)
            x = self.attn_norm(q + self.drop(attn))
            x = self.ffn_norm(x + self.drop(self.ffn(x)))

            mask = hotword_token_mask[:, start:end].reshape(bsz * cur_k, n_tokens).to(x.device)
            weight = mask.unsqueeze(-1).float()
            pooled = (x * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
            cur_logits = self.classifier(pooled).view(bsz, cur_k)
            logits.append(cur_logits)

        out = torch.cat(logits, dim=1)
        if hotword_valid_mask is not None:
            out = out.masked_fill(~hotword_valid_mask.to(out.device), -30.0)
        return out

    def save(self, path: str, extra: Dict = None) -> None:
        torch.save({"config": self.config(), "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu"):
        payload = torch.load(path, map_location=map_location)
        model = cls(**payload["config"])
        model.load_state_dict(payload["state_dict"], strict=True)
        return model, payload.get("extra", {})
