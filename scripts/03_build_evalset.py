"""03 构建评测集 → data/processed/evalset.jsonl

用法：
    python scripts/03_build_evalset.py 150      # 抽 150 篇政策，每篇生成 2 问
    python scripts/03_build_evalset.py 40 1     # 小规模试跑
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxrag.config import CORPUS_JSONL, EVALSET_JSONL  # noqa: E402
from taxrag.eval.dataset import generate_for_doc, sample_docs  # noqa: E402
from taxrag.io_utils import Progress, read_jsonl, write_jsonl  # noqa: E402
from taxrag.llm import DeepSeekChat  # noqa: E402


def main() -> None:
    n_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    per_doc = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    docs = read_jsonl(CORPUS_JSONL)
    if not docs:
        raise SystemExit(f"未找到语料：{CORPUS_JSONL}")

    sampled = sample_docs(docs, n_docs)
    print(f"从 {len(docs)} 篇中分层抽样 {len(sampled)} 篇，每篇生成 {per_doc} 问")

    client = DeepSeekChat()
    items: list[dict] = []
    bar = Progress(len(sampled), label="生成问题")
    for i, doc in enumerate(sampled, 1):
        items.extend(generate_for_doc(doc, client, n=per_doc))
        bar.tick(i)
    bar.done()

    for j, it in enumerate(items, 1):
        it["qid"] = f"q{j:04d}"
    write_jsonl(EVALSET_JSONL, items)

    print(f"\n评测集 {len(items)} 条 -> {EVALSET_JSONL}")
    print(f"  覆盖政策 {len({i['gold_doc_id'] for i in items})} 篇")
    print(f"  LLM 用量：{client.usage}")
    print("\n样例：")
    for it in items[:3]:
        print(f"  Q {it['query']}")
        print(f"    金标 {it['gold_doc_no'] or it['gold_doc_id']} | {it['gold_title'][:40]}")


if __name__ == "__main__":
    main()
