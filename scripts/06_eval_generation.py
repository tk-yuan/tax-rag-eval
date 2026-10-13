"""06 生成层评测：答案忠实度（faithfulness）与拒答率

检索对了不等于答得对。这一步专门测生成层：
  - 从评测集抽样提问 → 走完整 RAG 链路生成答案
  - 用另一个 LLM 当裁判，判断答案是否被检索片段支撑
  - 同时统计拒答率（检索不到时应当拒答，而不是硬编）

用法：
    python scripts/06_eval_generation.py 30      # 抽 30 条
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxrag.config import EMBED_MODEL, EVALSET_JSONL, PROCESSED_DIR, REPORT_DIR  # noqa: E402
from taxrag.eval.faithfulness import judge_faithfulness  # noqa: E402
from taxrag.io_utils import Progress, read_jsonl  # noqa: E402
from taxrag.llm import DeepSeekChat  # noqa: E402
from taxrag.rag import PolicyRAG  # noqa: E402
from taxrag.retrieve import Retriever  # noqa: E402


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    evalset = read_jsonl(EVALSET_JSONL)[:n]
    if not evalset:
        raise SystemExit("未找到评测集，请先运行 03_build_evalset.py")

    chunks = read_jsonl(PROCESSED_DIR / "chunks_article.jsonl")
    retriever = Retriever(chunks, "article", EMBED_MODEL, mode="hybrid", use_rerank=True)
    llm = DeepSeekChat()
    rag = PolicyRAG(retriever, llm=llm)

    print(f"对 {len(evalset)} 条 query 走完整 RAG 链路，并用 LLM 裁判打分\n")

    rows = []
    bar = Progress(len(evalset), label="评测")
    for i, item in enumerate(evalset, 1):
        r = rag.answer(item["query"])
        hits = retriever.search(item["query"])
        verdict = (
            {"supported": True, "score": 1.0, "unsupported_claims": []}
            if r["refused"]
            else judge_faithfulness(item["query"], r["answer"], hits, llm)
        )
        rows.append(
            {
                "qid": item["qid"],
                "query": item["query"],
                "answer": r["answer"],
                "refused": r["refused"],
                "faithfulness": verdict["score"],
                "supported": verdict["supported"],
                "unsupported_claims": verdict["unsupported_claims"],
                "gold_doc_no": item.get("gold_doc_no", ""),
            }
        )
        bar.tick(i)
    bar.done()

    n_rows = len(rows)
    faith = sum(r["faithfulness"] for r in rows) / n_rows
    refused = sum(1 for r in rows if r["refused"])
    bad = [r for r in rows if not r["supported"] and not r["refused"]]

    summary = {
        "n": n_rows,
        "faithfulness_mean": round(faith, 4),
        "supported_rate": round(sum(1 for r in rows if r["supported"]) / n_rows, 4),
        "refusal_rate": round(refused / n_rows, 4),
        "unsupported_cases": len(bad),
        "llm_usage": llm.usage,
    }

    (REPORT_DIR / "generation_eval.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n忠实度均值     {faith:.3f}")
    print(f"有依据比例     {summary['supported_rate']:.1%}")
    print(f"拒答率         {summary['refusal_rate']:.1%}")
    print(f"无依据且未拒答 {len(bad)} 条")
    print(f"LLM 用量       {llm.usage}")
    print(f"\n明细 -> {REPORT_DIR / 'generation_eval.json'}")

    if bad:
        print("\n无依据的答案抽样（需人工复核，LLM 裁判本身有噪声）：")
        for r in bad[:5]:
            print(f"  Q {r['query'][:50]}")
            print(f"    A {r['answer'][:90]}")
            for c in r["unsupported_claims"][:2]:
                print(f"    x {c[:80]}")


if __name__ == "__main__":
    main()
