"""向量索引的测试：检索正确性 + 增量更新。

**增量更新这块必须测**：政策库是持续变动的，更新逻辑写错的话，
要么新政策检索不到（漏召），要么旧政策删不掉（召回作废依据）。
"""

import numpy as np
import pytest

from taxrag.index.store import VectorStore


def _store(n: int = 5, dim: int = 8) -> VectorStore:
    rng = np.random.default_rng(42)
    ids = [f"doc{i}:0000" for i in range(n)]
    metas = [{"chunk_id": cid, "doc_id": cid.split(":")[0]} for cid in ids]
    return VectorStore(ids, metas, rng.random((n, dim)).tolist())


# —— 基础检索 ——
def test_search_returns_k_results():
    s = _store()
    q = np.ones(8).tolist()
    assert len(s.search(q, 3)) == 3


def test_search_scores_descending():
    s = _store()
    hits = s.search(np.ones(8).tolist(), 5)
    scores = [sc for _, sc in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_k_larger_than_corpus():
    s = _store(3)
    assert len(s.search(np.ones(8).tolist(), 10)) == 3


def test_search_query_is_normalized():
    """查询向量归一化后，点积即余弦相似度；缩放查询不应改变排序。"""
    s = _store()
    a = [i for i, _ in s.search(np.ones(8).tolist(), 3)]
    b = [i for i, _ in s.search((np.ones(8) * 7.5).tolist(), 3)]
    assert a == b


# —— 增量更新：新增 ——
def test_upsert_appends_new_ids():
    s = _store(3)
    r = s.upsert(["new:0000"], [{"chunk_id": "new:0000", "doc_id": "new"}],
                 np.ones((1, 8)).tolist())
    assert r["added"] == 1 and r["updated"] == 0
    assert len(s) == 4
    assert "new:0000" in s.ids


def test_upsert_updates_existing_in_place():
    s = _store(3)
    before = len(s)
    r = s.upsert(["doc1:0000"], [{"chunk_id": "doc1:0000", "doc_id": "doc1",
                                  "title": "改过的标题"}],
                 np.ones((1, 8)).tolist())
    assert r["added"] == 0 and r["updated"] == 1
    assert len(s) == before, "更新不应改变规模"
    idx = s.ids.index("doc1:0000")
    assert s.metadatas[idx]["title"] == "改过的标题"


def test_upsert_mixed_add_and_update():
    s = _store(3)
    r = s.upsert(
        ["doc0:0000", "brand_new:0000"],
        [{"doc_id": "doc0"}, {"doc_id": "brand_new"}],
        np.ones((2, 8)).tolist(),
    )
    assert r["added"] == 1 and r["updated"] == 1 and r["total"] == 4


def test_upsert_empty_is_noop():
    s = _store(3)
    assert s.upsert([], [], [])["total"] == 3


def test_upsert_normalizes_new_vectors():
    """新增向量也要归一化，否则与查询的量纲不一致，相似度会算错。"""
    s = _store(2)
    s.upsert(["big:0000"], [{"doc_id": "big"}], (np.ones((1, 8)) * 100).tolist())
    norms = np.linalg.norm(s.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_upserted_doc_is_findable():
    """新增的片段必须真的能被检索到——这是增量更新存在的意义。"""
    s = _store(3)
    target = np.zeros(8)
    target[0] = 1.0
    s.upsert(["target:0000"], [{"chunk_id": "target:0000", "doc_id": "target"}],
             target.reshape(1, -1).tolist())
    hits = s.search(target.tolist(), 1)
    assert s.ids[hits[0][0]] == "target:0000"


# —— 增量更新：删除 ——
def test_remove_drops_ids():
    s = _store(5)
    assert s.remove(["doc1:0000", "doc2:0000"]) == 2
    assert len(s) == 3
    assert "doc1:0000" not in s.ids


def test_remove_unknown_id_is_safe():
    s = _store(3)
    assert s.remove(["nope:0000"]) == 0
    assert len(s) == 3


def test_remove_empty_list():
    s = _store(3)
    assert s.remove([]) == 0


def test_removed_doc_no_longer_retrievable():
    """删掉的片段不能再被召回——否则检索层会继续拿作废依据作答。"""
    s = _store(3)
    q = s.vectors[1].copy()
    assert s.ids[s.search(q.tolist(), 1)[0][0]] == "doc1:0000"
    s.remove(["doc1:0000"])
    hits = s.search(q.tolist(), 3)
    assert all(s.ids[i] != "doc1:0000" for i, _ in hits)


def test_remove_keeps_vectors_aligned():
    """删除后 ids / metadatas / vectors 三者必须仍一一对应。"""
    s = _store(6)
    s.remove(["doc0:0000", "doc3:0000"])
    assert len(s.ids) == len(s.metadatas) == len(s.vectors) == 4
    for i, cid in enumerate(s.ids):
        assert s.metadatas[i]["doc_id"] == cid.split(":")[0]


# —— 统计与持久化 ——
def test_stats_counts_docs_not_chunks():
    rng = np.random.default_rng(0)
    s = VectorStore(
        ["d1:0000", "d1:0001", "d2:0000"],
        [{"doc_id": "d1"}, {"doc_id": "d1"}, {"doc_id": "d2"}],
        rng.random((3, 8)).tolist(),
    )
    st = s.stats()
    assert st["chunks"] == 3 and st["docs"] == 2 and st["dim"] == 8


def test_save_load_roundtrip(tmp_path):
    s = _store(4)
    s.save(tmp_path)
    again = VectorStore.load(tmp_path)
    assert again.ids == s.ids
    assert again.vectors.shape == s.vectors.shape
    np.testing.assert_allclose(again.vectors, s.vectors)


def test_upsert_then_save_load_keeps_new_doc(tmp_path):
    s = _store(2)
    s.upsert(["added:0000"], [{"doc_id": "added"}], np.ones((1, 8)).tolist())
    s.save(tmp_path)
    assert "added:0000" in VectorStore.load(tmp_path).ids
