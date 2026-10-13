"""01 清洗政策原文 → data/processed/corpus.jsonl

用法：
    python scripts/01_clean_corpus.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxrag.config import CORPUS_JSONL, RAW_DIR  # noqa: E402
from taxrag.corpus.clean import clean_document  # noqa: E402


def main() -> None:
    files = sorted(RAW_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"未找到原始语料：{RAW_DIR}")

    records = [clean_document(p) for p in files]
    CORPUS_JSONL.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )

    n = len(records)
    print(f"清洗完成：{n} 篇 -> {CORPUS_JSONL}")
    print(
        f"  抽到文号 {sum(1 for r in records if r['doc_no'])}/{n}"
        f" | 发布日期 {sum(1 for r in records if r['issued'])}/{n}"
        f" | 执行截止日 {sum(1 for r in records if r['valid_to'])}/{n}"
    )
    avg_keep = sum(r["kept_ratio"] for r in records) / n
    avg_chars = sum(r["chars"] for r in records) / n
    print(f"  平均正文 {avg_chars:.0f} 字 | 平均去噪比例 {1 - avg_keep:.1%}")

    years = Counter((r["issued"] or "????")[:4] for r in records)
    print("  按年份发文量 top8:", dict(sorted(years.items(), key=lambda kv: -kv[1])[:8]))

    print("\n样例：")
    r = records[0]
    print(f"  标题   {r['title']}")
    print(f"  文号   {r['doc_no']}")
    print(f"  发布   {r['issued']} | 发文机关 {r['issuer']} | 执行至 {r['valid_to'] or '—'}")
    print(f"  正文   {r['text'][:120]}...")


if __name__ == "__main__":
    main()
