"""检索指标的单元测试。

指标算错的话，整份评测报告都是错的——所以这是最该有测试的地方。
"""

from taxrag.eval.metrics import (
    aggregate,
    evaluate_ranking,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_hit():
    assert recall_at_k(["a", "b", "c"], ["b"], 3) == 1.0


def test_recall_at_k_miss_outside_window():
    assert recall_at_k(["a", "b", "c"], ["b"], 2) == 1.0
    assert recall_at_k(["a", "x", "b"], ["b"], 2) == 0.0


def test_recall_dedups_repeated_doc():
    # 同一份政策的多个片段只应算一次，否则 Recall 会被重复项虚高
    assert recall_at_k(["a", "a", "a"], ["a"], 3) == 1.0
    assert recall_at_k(["a", "a", "b"], ["b"], 2) == 0.0


def test_mrr_first_position():
    assert mrr(["a", "b"], ["a"]) == 1.0


def test_mrr_second_position():
    assert mrr(["x", "a"], ["a"]) == 0.5


def test_mrr_no_hit():
    assert mrr(["x", "y"], ["a"]) == 0.0


def test_mrr_uses_first_hit_only():
    assert mrr(["x", "a", "a"], ["a"]) == 0.5


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == 1.0


def test_ndcg_penalises_lower_position():
    high = ndcg_at_k(["a", "x"], ["a"], 2)
    low = ndcg_at_k(["x", "a"], ["a"], 2)
    assert high > low


def test_ndcg_zero_when_no_hit():
    assert ndcg_at_k(["x", "y"], ["a"], 2) == 0.0


def test_precision_counts_noise():
    # 前 3 条里 1 条相关 -> 1/3
    assert round(precision_at_k(["a", "x", "y"], ["a"], 3), 4) == round(1 / 3, 4)


def test_evaluate_ranking_shape():
    r = evaluate_ranking(["a", "x"], ["a"], k=2)
    assert set(r) == {"recall@2", "hit@2", "mrr", "ndcg@2", "precision@2"}


def test_aggregate_averages():
    rows = [{"mrr": 1.0}, {"mrr": 0.0}]
    assert aggregate(rows) == {"mrr": 0.5}
