# qwen_asr/joint/hotword.py
"""热词召回：用粗识别文本做字符相似度检索。"""
from typing import List


class HotwordRetriever:
    """字符相似度热词召回。"""

    def __init__(self, hotwords: List[str], scorer: str = "fuzz"):
        self.hotwords = [h.strip() for h in hotwords if h.strip()]
        self.scorer = scorer
        if scorer == "fuzz":
            try:
                from rapidfuzz import process, fuzz
                self._process = process
                self._fuzz = fuzz
            except ImportError:
                print("未安装 rapidfuzz，热词检索改用编辑距离")
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
            # 粗识别通常只包含热词的一部分，用 partial_ratio 更合适。
            results = self._process.extract(
                query, self.hotwords, scorer=self._fuzz.partial_ratio, limit=topk
            )
            return [w for w, score, _ in results if score > 60]
        else:
            scored = [(w, self._ed.eval(query, w)) for w in self.hotwords]
            scored.sort(key=lambda x: x[1])
            return [w for w, _ in scored[:topk]]
