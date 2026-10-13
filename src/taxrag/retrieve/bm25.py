"""BM25 关键词检索：与向量检索互补的那一半。

两者为什么都要有：
  向量检索擅长语义相近——「小微企业优惠」能召回「小型微利企业减免」；
  但它会把文号、数字、专有名词这类「必须精确命中」的信号糊掉。
  BM25 恰好相反：字面不重合就不给分。

中文没有空格，这里优先用 jieba 分词；没装就退化为字符二元组，
不为一个分词器给项目加硬依赖。

BM25 是 Okapi 公式，本文件手写实现（不引第三方库），公式如下：
    score(q, d) = Σ_t IDF(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1-b+b·|d|/avgdl) )
    IDF(t) = ln( (N - n_t + 0.5) / (n_t + 0.5) + 1 )
"""

from __future__ import annotations

import math
import re
from collections import Counter

try:  # 有 jieba 就用真分词
    import jieba

    def tokenize(text: str) -> list[str]:
        return [w for w in jieba.lcut(text) if w.strip()]

    TOKENIZER_NAME = "jieba"

except ImportError:  # 没有就退化为字符二元组

    def tokenize(text: str) -> list[str]:
        text = re.sub(r"\s+", "", text)
        chars = [c for c in text if c.strip()]
        if len(chars) < 2:
            return chars
        return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]

    TOKENIZER_NAME = "char_bigram"


class BM25:
    """Okapi BM25，内存索引。语料规模（千级片段）下无需倒排优化。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._avgdl = 0.0

    def fit(self, corpus: list[str]) -> "BM25":
        self._docs = [tokenize(t) for t in corpus]
        self._tf = [Counter(d) for d in self._docs]
        self._df = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())
        n = len(self._docs)
        self._avgdl = sum(len(d) for d in self._docs) / n if n else 0.0
        return self

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        nt = self._df.get(term, 0)
        return math.log((n - nt + 0.5) / (nt + 0.5) + 1.0)

    def score(self, query: str, idx: int) -> float:
        tf = self._tf[idx]
        dl = len(self._docs[idx]) or 1
        s = 0.0
        for term in tokenize(query):
            f = tf.get(term, 0)
            if not f:
                continue
            s += self._idf(term) * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
            )
        return s

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """返回 [(文档下标, 分数), ...]，按分数降序。"""
        scored = [(i, self.score(query, i)) for i in range(len(self._docs))]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
