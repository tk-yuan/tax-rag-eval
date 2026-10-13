"""消融运行器：一组检索配置跑一遍评测集，产出指标 + 延迟 + API 成本。

设计原则：任何一组配置的结论，都必须能和基线对照着看。
所以这里同时记录三样东西——效果（检索指标）、代价（延迟、API 调用次数）、
以及逐条明细（便于事后做失败样本归因）。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from ..config import EMBED_MODEL, TOP_K_RECALL, TOP_K_RERANK
from ..index.embedder import reset_stats, snapshot_stats
from ..retrieve import Retriever
from .metrics import aggregate, evaluate_ranking


@dataclass(frozen=True)
class RunConfig:
    """一组消融配置。"""

    strategy: str = "article"
    mode: str = "hybrid"          # vector | bm25 | hybrid
    use_rerank: bool = True
    top_k_recall: int = TOP_K_RECALL
    top_k_final: int = TOP_K_RERANK

    @property
    def name(self) -> str:
        return f"{self.strategy}__{self.mode}__{'rerank' if self.use_rerank else 'raw'}"


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * p), len(s) - 1)
    return round(s[idx], 4)


def run_config(
    cfg: RunConfig,
    chunks: list[dict],
    evalset: list[dict],
    k: int = 3,
    ks: tuple[int, ...] = (1, 3),
    limit: int | None = None,
    retriever: Retriever | None = None,
    use_query_cache: bool = False,
) -> dict:
    """跑完整个评测集，返回汇总指标与明细。

    同一次检索同时算多个 k 的指标（Top-1 看「第一发命中」，Top-k 看「召回覆盖」），
    避免为了多口径重复调用 API。

    use_query_cache 默认 **False**：测延迟与成本时若命中缓存，
    耗时和调用数都会被压成 0，报出来的成本口径就是假的。
    """
    items = evalset[:limit] if limit else evalset
    if retriever is None:
        retriever = Retriever(
            chunks,
            cfg.strategy,
            EMBED_MODEL,
            mode=cfg.mode,
            use_rerank=cfg.use_rerank,
            top_k_recall=cfg.top_k_recall,
            top_k_final=max(cfg.top_k_final, max(ks)),
            use_query_cache=use_query_cache,
        )

    reset_stats()
    per_k: dict[int, list[dict]] = {kk: [] for kk in ks}
    latencies: list[float] = []
    misses: list[dict] = []

    for item in items:
        t0 = time.perf_counter()
        hits = retriever.search(item["query"])
        latencies.append(time.perf_counter() - t0)

        retrieved = [h["doc_id"] for h in hits]
        gold = [item["gold_doc_id"]]
        for kk in ks:
            per_k[kk].append(evaluate_ranking(retrieved, gold, k=kk))

        if evaluate_ranking(retrieved, gold, k=max(ks))[f"hit@{max(ks)}"] == 0.0:
            misses.append(
                {
                    "query": item["query"],
                    "gold_doc_no": item.get("gold_doc_no", ""),
                    "gold_title": item.get("gold_title", "")[:60],
                    "retrieved_titles": [h["title"][:40] for h in hits],
                }
            )

    stats = snapshot_stats()
    n = max(len(items), 1)
    return {
        "name": cfg.name,
        "config": asdict(cfg),
        "embed_model": EMBED_MODEL,
        "n_queries": len(items),
        "k": k,
        "metrics_by_k": {f"k={kk}": aggregate(per_k[kk]) for kk in ks},
        "metrics": aggregate(per_k[max(ks)]),
        "latency_sec": {
            "p50": _pct(latencies, 0.50),
            "p95": _pct(latencies, 0.95),
            "mean": round(sum(latencies) / n, 4),
        },
        "api_per_query": {
            "embed_calls": round(stats["embed_calls"] / n, 3),
            "rerank_calls": round(stats["rerank_calls"] / n, 3),
            "rerank_docs": round(stats["rerank_docs"] / n, 2),
        },
        "misses": misses,
        "per_query": per_k[max(ks)],
    }
