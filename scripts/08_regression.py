"""08 回归评测 —— 防止改动让检索质量悄悄退化

用法：
    python scripts/08_regression.py --save-baseline   # 首次：把当前成绩存成基线
    python scripts/08_regression.py                   # 之后：对比基线，退化就退出码非 0

## 为什么必须有回归

Agent 和 RAG 都是概率系统。你改了一处切分逻辑、换了个分块参数、
调了下提示词——单测全绿，但**检索质量可能已经掉了 5 个点**，
而你不会知道。没有回归集，就只能靠"感觉还行"。

## 回归集里放什么

三类，都要有：

  1. **指标基线** —— Recall@k / MRR / nDCG 不能掉超过容差
  2. **已知失败样本** —— 现在是"已知召不回"的那批。它们要是变了
     （变好或变差），都得有人看一眼，说明行为变了
  3. **金标准样例** —— 极少数必须永远答对的典型问题

容差默认 2 个百分点：不是 0，因为 embedding 接口本身有微小波动；
但也不该大到掩盖真实退化。
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

BASELINE = REPORT_DIR / "regression_baseline.json"
TOLERANCE_PP = 0.02      # 允许的退化幅度（百分点）
REGRESSION_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 80

# 回归只盯推荐配置：它是最终对外的那一套，其它配置的对比由 04 消融负责
CFG = RunConfig("article", "hybrid", True)
TRACKED = ("recall@1", "recall@3", "mrr", "ndcg@3", "precision@3")


def flat_metrics(res: dict) -> dict:
    """把按 k 分组存放的指标摊平。

    run_config 返回的 metrics_by_k 是 {"k=1": {...}, "k=3": {...}}，
    顶层 metrics 只有最大 k 的那一组。要拿 recall@1 必须去 k=1 那组拿——
    这里踩过一次 KeyError，别再踩。
    """
    m1 = res["metrics_by_k"]["k=1"]
    m3 = res["metrics_by_k"]["k=3"]
    return {
        "recall@1": m1["recall@1"],
        "recall@3": m3["recall@3"],
        "mrr": m3["mrr"],
        "ndcg@3": m3["ndcg@3"],
        "precision@3": m3["precision@3"],
    }


def _run() -> dict:
    chunks = read_jsonl(PROCESSED_DIR / f"chunks_{CFG.strategy}.jsonl")
    if not chunks:
        raise SystemExit("缺切分结果，请先运行 02_build_index.py")
    if not index_exists(CFG.strategy, EMBED_MODEL):
        raise SystemExit("缺向量索引，请先运行 02_build_index.py")
    evalset = read_jsonl(EVALSET_JSONL)
    if not evalset:
        raise SystemExit("缺评测集，请先运行 03_build_evalset.py")
    return run_config(CFG, chunks, evalset, limit=REGRESSION_LIMIT)


def cmd_save() -> None:
    res = _run()
    metrics = flat_metrics(res)
    payload = {
        "config": res["config"],
        "n_queries": res["n_queries"],
        "metrics": metrics,
        "latency_p50": res["latency_sec"]["p50"],
        "known_misses": [
            {"query": m["query"], "gold_doc_no": m["gold_doc_no"]}
            for m in res["misses"]
        ],
    }
    BASELINE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"基线已保存 → {BASELINE}")
    print(f"  配置 {CFG.name} | {res['n_queries']} 条 query | {len(payload['known_misses'])} 条已知失败样本")
    for k in TRACKED:
        print(f"  {k:12s} {metrics[k]:.4f}")


def cmd_check() -> int:
    if not BASELINE.exists():
        print("还没有基线。先跑：python scripts/08_regression.py --save-baseline")
        return 2
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    res = _run()
    metrics = flat_metrics(res)

    print(f"配置 {CFG.name} | {res['n_queries']} 条 query（基线 {base['n_queries']} 条）\n")
    failed: list[str] = []
    print(f"{'指标':<14}{'基线':>9}{'本次':>9}{'变化':>10}   判定")
    for k in TRACKED:
        b, now = base["metrics"][k], metrics[k]
        delta = now - b
        if delta < -TOLERANCE_PP:
            verdict = "✗ 退化"
            failed.append(f"{k} 从 {b:.4f} 掉到 {now:.4f}（{delta*100:+.2f}pp）")
        elif delta > TOLERANCE_PP:
            verdict = "↑ 提升"
        else:
            verdict = "✓"
        print(f"{k:<14}{b:>9.4f}{now:>9.4f}{delta*100:>+9.2f}pp   {verdict}")

    # 已知失败样本：变化就要有人看一眼
    base_miss = {m["query"] for m in base.get("known_misses", [])}
    now_miss = {m["query"] for m in res["misses"]}
    fixed = base_miss - now_miss
    broke = now_miss - base_miss

    print(f"\n已知失败样本 {len(base_miss)} 条 → 现在 {len(now_miss)} 条")
    if fixed:
        print(f"  ↑ {len(fixed)} 条已经能召回了（改好了，值得确认是不是有意的）：")
        for q in list(fixed)[:3]:
            print(f"      {q[:56]}")
    if broke:
        print(f"  ✗ {len(broke)} 条新出现的失败（这是回归）：")
        for q in list(broke)[:5]:
            print(f"      {q[:56]}")
        failed.extend([f"新失败样本：{q}" for q in list(broke)[:3]])

    print()
    if failed:
        print("=== 回归失败 ===")
        for f in failed:
            print(f"  · {f}")
        print("\n要么修回原状，要么确认新行为是对的、再用 --save-baseline 更新基线。")
        return 1
    print("=== 回归通过 ===")
    return 0


def main() -> None:
    if "--save-baseline" in sys.argv:
        cmd_save()
    else:
        raise SystemExit(cmd_check())


if __name__ == "__main__":
    main()
