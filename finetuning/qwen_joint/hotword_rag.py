# qwen_joint/hotword_rag.py
"""简单热词 RAG：基于字符 n-gram / 编辑距离 / 可选向量检索。
默认用 rapidfuzz（纯 CPU，快），如未安装退化到 editdistance。
"""
from typing import List, Optional


class HotwordRetriever:
    """基础检索器 —— 字符相似度召回。适合短词库（万级）。"""

    def __init__(self, hotwords: List[str], scorer: str = "fuzz"):
        self.hotwords = [h.strip() for h in hotwords if h.strip()]
        self.scorer = scorer
        if scorer == "fuzz":
            try:
                from rapidfuzz import process, fuzz
                self._process = process
                self._fuzz = fuzz
            except ImportError:
                print("[HotwordRetriever] rapidfuzz 未安装，退化到 editdistance")
                self.scorer = "edit"
        if self.scorer == "edit":
            import editdistance
            self._ed = editdistance

    @classmethod
    def from_file(cls, path: str, **kwargs):
        with open(path, "r", encoding="utf-8") as f:
            hotwords = [line.strip() for line in f if line.strip()]
        return cls(hotwords, **kwargs)

    def retrieve(self, query: str, topk: int = 10) -> List[str]:
        if not self.hotwords or not query:
            return []
        if self.scorer == "fuzz":
            # partial_ratio 更适合"粗识别包含热词子串"的情况
            results = self._process.extract(
                query, self.hotwords, scorer=self._fuzz.partial_ratio, limit=topk
            )
            return [w for w, score, _ in results if score > 60]
        else:
            scored = [(w, self._ed.eval(query, w)) for w in self.hotwords]
            scored.sort(key=lambda x: x[1])
            return [w for w, _ in scored[:topk]]


class EmbeddingHotwordRetriever:
    """向量检索版本（需要 sentence-transformers + faiss）。可选。"""

    def __init__(self, hotwords: List[str], model_name: str = "BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.hotwords = hotwords
        self.model = SentenceTransformer(model_name)
        self.embs = self.model.encode(hotwords, normalize_embeddings=True)
        self.np = np

    def retrieve(self, query: str, topk: int = 10) -> List[str]:
        if not self.hotwords or not query:
            return []
        q = self.model.encode([query], normalize_embeddings=True)
        sims = (self.embs @ q.T).squeeze(-1)
        idx = self.np.argsort(-sims)[:topk]
        return [self.hotwords[i] for i in idx]
