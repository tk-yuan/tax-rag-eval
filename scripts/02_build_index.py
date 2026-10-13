"""02 切分 + 向量化 → chunks.jsonl 与 Chroma 索引

用法：
    python scripts/02_build_index.py                 # 三种策略全建
    python scripts/02_build_index.py article         # 只建某一种
    python scripts/02_build_index.py article --rebuild
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxrag.config import CHUNKS_JSONL, CORPUS_JSONL, EMBED_MODEL, PROCESSED_DIR  # noqa: E402
from taxrag.index.chunkers import STRATEGIES, chunk  # noqa: E402
from taxrag.index.store import build_index  # noqa: E402
from taxrag.io_utils import Progress, read_jsonl, write_jsonl  # noqa: E402


def load_corpus() -> list[dict]:
    docs = read_jsonl(CORPUS_JSONL)
    if not docs:
        raise SystemExit(f"未找到语料：{CORPUS_JSONL}，请先运行 01_clean_corpus.py")
    return docs


def build_chunks(docs: list[dict], strategy: str) -> list[dict]:
    """把每篇政策切成片段，带上可追溯的元数据。"""
    out: list[dict] = []
    bar = Progress(len(docs), label=f"{strategy:10s}")
    for i, doc in enumerate(docs, 1):
        for j, text in enumerate(chunk(doc["text"], strategy)):
            out.append(
                {
                    "chunk_id": f"{strategy}:{doc['doc_id']}:{j:04d}",
                    "doc_id": doc["doc_id"],
                    "chunk_index": j,
                    "text": text,
                    "chars": len(text),
                    "title": doc.get("title", ""),
                    "doc_no": doc.get("doc_no", ""),
                    "issued": doc.get("issued", ""),
                    "valid_to": doc.get("valid_to", ""),
                    "issuer": doc.get("issuer", ""),
                }
            )
        bar.tick(i)
    bar.done()
    return out


def chunks_path(strategy: str) -> Path:
    return PROCESSED_DIR / f"chunks_{strategy}.jsonl"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rebuild = "--rebuild" in sys.argv
    strategies = args or list(STRATEGIES)

    docs = load_corpus()
    print(f"语料 {len(docs)} 篇 | 向量模型 {EMBED_MODEL}")

    for strategy in strategies:
        chunks = build_chunks(docs, strategy)
        write_jsonl(chunks_path(strategy), chunks)
        avg = sum(c["chars"] for c in chunks) / len(chunks)
        print(f"  {strategy:10s} {len(chunks):5d} 片段 | 平均 {avg:.0f} 字")

        # 顺带把默认策略也写一份到 chunks.jsonl，方便下游脚本直接读
        if strategy == "article":
            write_jsonl(CHUNKS_JSONL, chunks)

        print(f"  {strategy:10s} 建立向量索引 …")
        build_index(chunks, strategy, EMBED_MODEL, rebuild=rebuild)
        print(f"  {strategy:10s} 索引就绪")

    print("\n完成。索引位于 .cache/index/（向量 .npy + 元数据 .json）")


if __name__ == "__main__":
    main()
