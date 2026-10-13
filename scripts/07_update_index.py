"""07 增量更新索引 —— 政策库变了，不必全量重建

用法：
    python scripts/07_update_index.py              # 所有已建策略
    python scripts/07_update_index.py article      # 只更新某一种
    python scripts/07_update_index.py --dry-run    # 只看差异，不写

## 为什么需要增量

政策是持续发布的。财政部每月都有新公告，一份新政策落库后：
  · **全量重建**：把几千条片段重新切分、重新 embed，几分钟 + 一笔 API 费用
  · **增量更新**：只处理新增和移除的那几条

代价也要说清楚：向量矩阵要重建一次（np.vstack），所以**批量更新比逐条更新划算**，
最好攒一批一起更，而不是来一条更一次。

## 怎么发现差异

以 `data/processed/corpus.jsonl` 为准（它是清洗脚本的产物，是"当前应有"的真相）：
  · 语料有、索引没有 → 新增
  · 索引有、语料没有 → 移除（政策被撤回或清理掉了）

注意这与"政策失效"不是一回事：政策到期应由检索层的 `valid_to` 过滤，
不是从索引里删掉——过期政策依然有历史检索价值。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxrag.config import CORPUS_JSONL, EMBED_MODEL  # noqa: E402
from taxrag.index.chunkers import chunk  # noqa: E402
from taxrag.index.embedder import SiliconFlowEmbeddings  # noqa: E402
from taxrag.index.store import index_exists, index_path, load_index  # noqa: E402
from taxrag.io_utils import read_jsonl  # noqa: E402


def plan_diff(strategy: str, docs: list[dict]) -> dict:
    """比较语料与索引，算出要加什么、要删什么。"""
    store = load_index(strategy, EMBED_MODEL)
    indexed_docs = {m.get("doc_id") for m in store.metadatas}
    corpus_docs = {d["doc_id"] for d in docs}
    return {
        "strategy": strategy,
        "indexed_chunks": len(store),
        "new_docs": sorted(corpus_docs - indexed_docs),
        "removed_docs": sorted(indexed_docs - corpus_docs),
    }


def apply_diff(strategy: str, docs: list[dict], diff: dict, dry_run: bool) -> dict:
    if dry_run or (not diff["new_docs"] and not diff["removed_docs"]):
        return {"added": 0, "updated": 0, "removed": 0, "total": diff["indexed_chunks"]}

    store = load_index(strategy, EMBED_MODEL)
    by_id = {d["doc_id"]: d for d in docs}

    # —— 新增 ——
    new_chunks: list[dict] = []
    for doc_id in diff["new_docs"]:
        doc = by_id[doc_id]
        for j, text in enumerate(chunk(doc["text"], strategy)):
            new_chunks.append({
                "chunk_id": f"{strategy}:{doc_id}:{j:04d}",
                "doc_id": doc_id,
                "chunk_index": j,
                "text": text,
                "title": doc.get("title", ""),
                "doc_no": doc.get("doc_no", ""),
                "issued": doc.get("issued", ""),
                "valid_to": doc.get("valid_to", ""),
                "issuer": doc.get("issuer", ""),
            })

    result = {"added": 0, "updated": 0, "removed": 0}
    if new_chunks:
        emb = SiliconFlowEmbeddings(model=EMBED_MODEL)
        vecs = emb.embed_documents([c["text"] for c in new_chunks])
        r = store.upsert(
            ids=[c["chunk_id"] for c in new_chunks],
            metadatas=[{k: c.get(k, "") for k in
                        ("chunk_id", "doc_id", "title", "doc_no", "issued",
                         "valid_to", "issuer", "chunk_index")} for c in new_chunks],
            vectors=vecs,
        )
        result.update(added=r["added"], updated=r["updated"])

    # —— 移除 ——
    if diff["removed_docs"]:
        stale = [
            cid for cid, m in zip(store.ids, store.metadatas)
            if m.get("doc_id") in set(diff["removed_docs"])
        ]
        result["removed"] = store.remove(stale)

    store.save(index_path(strategy, EMBED_MODEL))
    result["total"] = len(store)
    return result


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry_run = "--dry-run" in sys.argv
    strategies = args or ["article", "fixed", "recursive"]

    docs = read_jsonl(CORPUS_JSONL)
    if not docs:
        raise SystemExit(f"未找到语料：{CORPUS_JSONL}，请先运行 01_clean_corpus.py")
    print(f"语料 {len(docs)} 篇文档\n")

    touched = False
    for strategy in strategies:
        if not index_exists(strategy, EMBED_MODEL):
            print(f"  {strategy:10s} 跳过（索引不存在，先运行 02_build_index.py）")
            continue
        diff = plan_diff(strategy, docs)
        n_new, n_del = len(diff["new_docs"]), len(diff["removed_docs"])
        mark = "✓ 一致" if not (n_new or n_del) else f"新增 {n_new} 篇 / 移除 {n_del} 篇"
        print(f"  {strategy:10s} 现有 {diff['indexed_chunks']:5d} 片段 | {mark}")

        if n_new or n_del:
            touched = True
            if not dry_run:
                r = apply_diff(strategy, docs, diff, dry_run=False)
                print(f"  {'':10s} 已更新：+{r['added']} 更新 {r['updated']} 删除 {r['removed']}"
                      f" → 共 {r['total']} 片段")

    if dry_run:
        print("\n（--dry-run：只报差异，未写盘）")
    elif not touched:
        print("\n所有索引与语料一致，无需更新。")


if __name__ == "__main__":
    main()
