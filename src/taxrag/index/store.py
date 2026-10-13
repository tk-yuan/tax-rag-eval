"""向量存储：自实现的精确余弦检索。

## 为什么不用向量数据库

先说结论：**在本项目的规模下，向量库是负收益。**

过程：最初用 Chroma（`langchain_chroma`）。建索引时报
`Error loading hnsw index`——**81 条的小集合正常，上千条就崩**，且只在
**跨进程重新打开**时暴露（同进程内重读正常）。试过预计算向量、把写入批次从
64 调大到 512，仍然复现。这是 chromadb 1.5.9 Rust 持久化的缺陷，不是用法问题。

于是做了一次量化评估：

| 方案 | 单次查询耗时 | 依赖风险 |
|---|---|---|
| Chroma / FAISS（近似检索，HNSW/IVF） | 微秒级 | 有 |
| **numpy 精确检索（本实现）** | **0.26 ms**（2909×1024） | 无 |

而一次 embedding API 调用要几百毫秒——**检索耗时比它低三个数量级**，
近似索引省下的那点时间根本看不见。既然如此，引入一个会损坏、要额外安装的
依赖，纯属负收益。

规模真的上去之后（十万级以上片段）再换 FAISS/Milvus 不迟，
`VectorStore` 的接口就是为此留的。

## 数学

向量先归一化，点积即余弦相似度：
    cos(q, d) = q·d / (|q|·|d|) = q̂·d̂
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR
from .embedder import SiliconFlowEmbeddings

META_FIELDS = (
    "chunk_id",
    "doc_id",
    "title",
    "doc_no",
    "issued",
    "valid_to",
    "issuer",
    "chunk_index",
)


class VectorStore:
    """精确余弦检索。数据量在数千片段量级时，暴力扫全表就是最优解。"""

    def __init__(
        self, ids: list[str], metadatas: list[dict], vectors: list[list[float]]
    ) -> None:
        self.ids = ids
        self.metadatas = metadatas
        v = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        self.vectors = v / np.maximum(norms, 1e-9)

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query_vector: list[float] | np.ndarray, k: int) -> list[tuple[int, float]]:
        """返回 [(下标, 余弦相似度), ...]，按相似度降序。"""
        q = np.asarray(query_vector, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        sims = self.vectors @ q
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]      # O(N)，不必全排序
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    # —— 持久化：一个 .npy 存向量，一个 json 存元数据 ——
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        payload = {"ids": self.ids, "metadatas": self.metadatas}
        (path / "index.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        payload = json.loads((path / "index.json").read_text(encoding="utf-8"))
        vectors = np.load(path / "vectors.npy")
        obj = cls.__new__(cls)
        obj.ids = payload["ids"]
        obj.metadatas = payload["metadatas"]
        obj.vectors = vectors
        return obj


def collection_name(strategy: str, embed_model: str) -> str:
    safe = embed_model.replace("/", "_").replace("-", "_").lower()
    return f"policies_{strategy}_{safe}"


def index_path(strategy: str, embed_model: str, persist_dir: Path | None = None) -> Path:
    return Path(persist_dir or CACHE_DIR / "index") / collection_name(strategy, embed_model)


def build_index(
    chunks: list[dict],
    strategy: str,
    embed_model: str,
    persist_dir: Path | None = None,
    rebuild: bool = False,
) -> VectorStore:
    """切分结果 → 向量 → 落盘。已存在且规模一致时直接复用。"""
    path = index_path(strategy, embed_model, persist_dir)

    if not rebuild and path.exists():
        store = VectorStore.load(path)
        if len(store) == len(chunks):
            return store

    embedder = SiliconFlowEmbeddings(model=embed_model)
    vectors = embedder.embed_documents([c["text"] for c in chunks])
    store = VectorStore(
        ids=[c["chunk_id"] for c in chunks],
        metadatas=[{k: c.get(k, "") for k in META_FIELDS} for c in chunks],
        vectors=vectors,
    )
    store.save(path)
    return store


def load_index(
    strategy: str, embed_model: str, persist_dir: Path | None = None
) -> VectorStore:
    return VectorStore.load(index_path(strategy, embed_model, persist_dir))


def index_exists(strategy: str, embed_model: str, persist_dir: Path | None = None) -> bool:
    """索引是否已建好。

    调用方（如消融脚本）用它做前置检查，好在缺索引时给出「请先运行
    02_build_index.py <策略>」这种能照做的提示，而不是抛一个
    FileNotFoundError 让人去猜（这个坑本项目踩过）。
    """
    p = index_path(strategy, embed_model, persist_dir)
    return (p / "index.json").exists() and (p / "vectors.npy").exists()
