"""评测层：评测集构建、指标计算、消融运行。"""

from .metrics import aggregate, evaluate_ranking, mrr, ndcg_at_k, recall_at_k

__all__ = ["aggregate", "evaluate_ranking", "mrr", "ndcg_at_k", "recall_at_k"]
