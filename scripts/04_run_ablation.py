"""04 消融实验：切分策略 × 检索方式 × 是否重排

用法：
    python scripts/04_run_ablation.py            # 全量评测集
    python scripts/04_run_ablation.py 40         # 只用前 40 条（试跑）

产出：
    reports/ablation.json        原始结果
    reports/ablation_report.md   可读报告（含失败样本）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxrag.config import EMBED_MODEL, EVALSET_JSONL, PROCESSED_DIR, REPORT_DIR  # noqa: E402
from taxrag.eval.runner import RunConfig, run_config  # noqa: E402
from taxrag.index.store import index_exists  # noqa: E402
from taxrag.io_utils import read_jsonl  # noqa: E402

# 消融矩阵：每一项只改一个变量，其余固定，便于归因
#   —— 基线选「fixed + 纯向量 + 不重排」，即最朴素的 RAG 做法
ABLATION: list[RunConfig] = [
    RunConfig("fixed", "vector", False),      # 基线
    RunConfig("recursive", "vector", False),  # +切分策略
    RunConfig("article", "vector", False),    # +领域感知切分
    RunConfig("article", "bm25", False),      # 纯关键词
    RunConfig("article", "hybrid", False),    # +混合召回
    RunConfig("article", "vector", True),     # 纯向量 +重排
    RunConfig("article", "hybrid", True),     # 全量（推荐配置）
]

LABELS = {
    "fixed__vector__raw": "① 基线：固定切分 + 纯向量 + 不重排",
    "recursive__vector__raw": "② 换递归切分",
    "article__vector__raw": "③ 换条款切分",
    "article__bm25__raw": "④ 条款切分 + 纯 BM25",
    "article__hybrid__raw": "⑤ 条款切分 + 混合召回",
    "article__vector__rerank": "⑥ 条款切分 + 纯向量 + 重排",
    "article__hybrid__rerank": "⑦ 全量：条款切分 + 混合召回 + 重排",
}


def load_chunks(strategy: str) -> list[dict]:
    return read_jsonl(PROCESSED_DIR / f"chunks_{strategy}.jsonl")


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_report(results: list[dict], n_queries: int) -> str:
    lines: list[str] = []
    lines.append("# RAG 检索消融实验报告\n")
    lines.append(f"- 评测集：**{n_queries}** 条 query，金标为 doc 级（先截断 top-k 再去重）\n")
    lines.append(f"- 向量模型：`{results[0]['embed_model']}`｜重排：`BAAI/bge-reranker-v2-m3`\n")
    lines.append("- 指标：Recall@1 / Recall@3 / MRR / nDCG@3 / Precision@3\n")
    lines.append("- 成本：p50 延迟、每条 query 的 API 调用次数\n")

    def m1(r):
        return r["metrics_by_k"]["k=1"]

    def m3(r):
        return r["metrics_by_k"]["k=3"]

    lines.append("\n## 主结果\n")
    lines.append(
        "| 配置 | Recall@1 | Recall@3 | MRR | nDCG@3 | P@3 | 延迟 p50 | API/query |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        a, b = m1(r), m3(r)
        calls = r["api_per_query"]["embed_calls"] + r["api_per_query"]["rerank_calls"]
        lines.append(
            f"| {LABELS.get(r['name'], r['name'])} | {pct(a['recall@1'])} | {pct(b['recall@3'])} | "
            f"{b['mrr']:.3f} | {b['ndcg@3']:.3f} | {pct(b['precision@3'])} | "
            f"{r['latency_sec']['p50']:.2f}s | {calls:.2f} |"
        )

    lines.append("\n## 相对基线的增量\n")
    lines.append("| 配置 | Recall@3 变化 | MRR 变化 | 延迟变化 |")
    lines.append("|---|---|---|---|")
    base3, base1 = m3(results[0]), m1(results[0])
    for r in results[1:]:
        b = m3(r)
        lines.append(
            f"| {LABELS.get(r['name'], r['name'])} | "
            f"{(b['recall@3'] - base3['recall@3']) * 100:+.1f}pp | "
            f"{b['mrr'] - base3['mrr']:+.3f} | "
            f"{r['latency_sec']['p50'] - results[0]['latency_sec']['p50']:+.2f}s |"
        )
    _ = base1

    lines.append("\n## 失败样本（推荐配置下未命中）\n")
    best = results[-1]
    if not best["misses"]:
        lines.append("无。\n")
    else:
        lines.append(f"共 {len(best['misses'])} 条，抽样如下：\n")
        for m in best["misses"][:10]:
            lines.append(f"- **问**：{m['query']}")
            lines.append(f"  - 金标：{m['gold_doc_no'] or '—'} {m['gold_title']}")
            lines.append(f"  - 检索到：{'；'.join(m['retrieved_titles'][:3])}")
    return "\n".join(lines)


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    evalset = read_jsonl(EVALSET_JSONL)
    if not evalset:
        raise SystemExit(f"未找到评测集：{EVALSET_JSONL}，请先运行 03_build_evalset.py")

    n = limit or len(evalset)
    print(f"评测集 {len(evalset)} 条，本次跑 {n} 条\n")

    results: list[dict] = []
    skipped: list[str] = []
    for cfg in ABLATION:
        chunks = load_chunks(cfg.strategy)
        if not chunks:
            skipped.append(f"{cfg.strategy}（缺切分结果）")
            continue
        if not index_exists(cfg.strategy, EMBED_MODEL):
            skipped.append(f"{cfg.strategy}（缺向量索引）")
            continue
        print(f"运行 {LABELS.get(cfg.name, cfg.name)} …")
        res = run_config(cfg, chunks, evalset, limit=limit)
        m = res["metrics"]
        print(
            f"  recall@3={m['recall@3']:.3f} mrr={m['mrr']:.3f} "
            f"p50={res['latency_sec']['p50']:.2f}s"
        )
        results.append(res)

    if skipped:
        print("\n跳过的配置：")
        for s in sorted(set(skipped)):
            print(f"  · {s}")
        print("  补建索引：python scripts/02_build_index.py           # 三种策略一次建全")

    if not results:
        raise SystemExit(
            "\n没有任何配置跑成功。请先执行：\n"
            "  python scripts/01_clean_corpus.py\n"
            "  python scripts/02_build_index.py\n"
            "  python scripts/03_build_evalset.py 180"
        )
    if len(results) < len(ABLATION):
        print(f"\n注意：只跑了 {len(results)}/{len(ABLATION)} 组配置，"
              f"报告里的对照不完整。建议补齐索引后重跑。")

    (REPORT_DIR / "ablation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_report(results, n)
    (REPORT_DIR / "ablation_report.md").write_text(report, encoding="utf-8")

    print(f"\n结果 -> {REPORT_DIR / 'ablation.json'}")
    print(f"报告 -> {REPORT_DIR / 'ablation_report.md'}")


if __name__ == "__main__":
    main()
