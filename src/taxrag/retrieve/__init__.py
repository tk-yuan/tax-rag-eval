"""检索层：向量、BM25、混合召回与重排的统一入口。

Retriever 是消融实验的执行体——同一个 query，切换 mode / use_rerank，
就能得到不同的检索结果，供上层评测对比。
"""

from __future__ import annotations

from ..config import RRF_K, TOP_K_RECALL, TOP_K_RERANK
from ..index.embedder import SiliconFlowEmbeddings, SiliconFlowReranker
from ..index.store import load_index
from .bm25 import BM25

MODES = ("vector", "bm25", "hybrid")


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion：把多路排序合成一路。

    只看名次不看分数，所以不同量纲的检索器（余弦相似度 vs BM25 分）能安全融合。
    score(d) = Σ_r 1 / (k + rank_r(d))
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class Retriever:
    """政策检索器。

    mode:       vector | bm25 | hybrid（向量 + BM25 经 RRF 融合）
    use_rerank: 是否对召回结果做交叉编码重排
    """

    def __init__(
        self,
        chunks: list[dict],
        strategy: str,
        embed_model: str,
        mode: str = "hybrid",
        use_rerank: bool = True,
        top_k_recall: int = TOP_K_RECALL,
        top_k_final: int = TOP_K_RERANK,
        use_query_cache: bool = True,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"未知检索方式：{mode}（可选 {MODES}）")
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.mode = mode
        self.use_rerank = use_rerank
        self.top_k_recall = top_k_recall
        self.top_k_final = top_k_final

        self._store = None if mode == "bm25" else load_index(strategy, embed_model)
        # 查询向量缓存：线上是好事（同一问题重复问不用重算），
        # 但**测延迟和成本时必须关掉**——否则缓存命中会把耗时和调用数
        # 都压成 0，报告出来的成本口径就失真了。
        self._embedder = (
            None
            if mode == "bm25"
            else SiliconFlowEmbeddings(model=embed_model, use_cache=use_query_cache)
        )
        self._bm25 = BM25().fit([c["text"] for c in chunks]) if mode != "vector" else None
        self._ranker = SiliconFlowReranker() if use_rerank else None
        self._ids = [c["chunk_id"] for c in chunks]

    # —— 各路召回 ——
    def _vector(self, query: str, k: int) -> list[str]:
        qv = self._embedder.embed_query(query)
        return [self._store.ids[i] for i, _ in self._store.search(qv, k)]

    def _keyword(self, query: str, k: int) -> list[str]:
        return [self._ids[i] for i, _ in self._bm25.search(query, top_k=k)]

    def search(self, query: str) -> list[dict]:
        """返回精排后的最终片段：[{chunk_id, text, score, ...}, ...]"""
        k = self.top_k_recall
        if self.mode == "vector":
            recalled = self._vector(query, k)
        elif self.mode == "bm25":
            recalled = self._keyword(query, k)
        else:
            fused = rrf_fuse([self._vector(query, k), self._keyword(query, k)])
            recalled = [cid for cid, _ in fused[:k]]

        if self.use_rerank and recalled:
            docs = [self.by_id[cid]["text"] for cid in recalled]
            ranked = self._ranker.rerank(query, docs, top_n=self.top_k_final)
            ordered = [(recalled[r["index"]], r["score"]) for r in ranked]
        else:
            ordered = [(cid, 1.0 / (i + 1)) for i, cid in enumerate(recalled[: self.top_k_final])]

        out = []
        for rank, (cid, score) in enumerate(ordered, 1):
            c = dict(self.by_id[cid])
            c["score"] = score
            c["rank"] = rank
            out.append(c)
        return out
