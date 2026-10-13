"""检索与生成评测指标。

检索层（本项目的重点）：
    Recall@k        —— 前 k 条里是否命中金标（命中率）
    MRR             —— 第一个命中项的排名倒数（排得越前越好）
    nDCG@k          —— 考虑多个金标与排序位置的折损累积增益
    ContextPrecision@k —— 前 k 条里有多少是真正相关的（衡量「噪声」）

生成层（RAG Triad 的前两项，简化实现）：
    Faithfulness    —— 答案是否被检索到的片段支撑（不算无中生有）
    AnswerRelevance —— 答案是否切题

指标定义都按「doc 级金标」：一个 query 对应若干正确文档，检索结果以 doc_id 去重评判。
比 chunk 级更接近实际使用——用户要的是「找到对的那份政策」，不是「找到对的第 37 个字」。
"""

from __future__ import annotations

import math


def _dedup_docs(doc_ids: list[str]) -> list[str]:
    seen, out = set(), []
    for d in doc_ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


# 口径说明：先按名次截断 top-k，再把片段折叠到 doc 级。
# 顺序不能反——同一份政策的两个片段占据了第 1、2 名，
# 意味着它挤掉了另一个可能相关的政策，这个代价必须体现出来。
# 若先折叠再截断，等于白送一个名额，指标会虚高。


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """前 k 条里命中了多少比例的金标文档。"""
    if not gold:
        return 0.0
    top = set(_dedup_docs(retrieved[:k]))
    return len(top & set(gold)) / len(set(gold))


def hit_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """前 k 条里是否至少命中一个金标（0/1）。"""
    return 1.0 if set(_dedup_docs(retrieved[:k])) & set(gold) else 0.0


def mrr(retrieved: list[str], gold: list[str]) -> float:
    """第一个命中项的排名倒数。"""
    gold_set = set(gold)
    for i, d in enumerate(_dedup_docs(retrieved), 1):
        if d in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """二值相关性的 nDCG@k。"""
    if not gold:
        return 0.0
    gold_set = set(gold)
    docs = _dedup_docs(retrieved[:k])
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(docs, 1) if d in gold_set)
    ideal_n = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """前 k 条里相关文档占比——衡量检索结果里的噪声。"""
    docs = _dedup_docs(retrieved[:k])
    if not docs:
        return 0.0
    return len(set(docs) & set(gold)) / len(docs)


def evaluate_ranking(retrieved: list[str], gold: list[str], k: int = 3) -> dict:
    """一次检索的完整指标。"""
    return {
        f"recall@{k}": recall_at_k(retrieved, gold, k),
        f"hit@{k}": hit_at_k(retrieved, gold, k),
        "mrr": mrr(retrieved, gold),
        f"ndcg@{k}": ndcg_at_k(retrieved, gold, k),
        f"precision@{k}": precision_at_k(retrieved, gold, k),
    }


def aggregate(rows: list[dict]) -> dict:
    """对多条 query 的指标求平均。"""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
