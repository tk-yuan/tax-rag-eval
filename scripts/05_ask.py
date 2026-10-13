"""05 提问演示：检索 + 生成带引用的答案

用法：
    python scripts/05_ask.py "我们公司买了一台 400 万的设备，能一次性扣除吗"
    python scripts/05_ask.py                      # 进入交互模式

    # 换配置
    python scripts/05_ask.py -s fixed -m vector --no-rerank "问题"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxrag.config import EMBED_MODEL, PROCESSED_DIR  # noqa: E402
from taxrag.io_utils import read_jsonl  # noqa: E402
from taxrag.rag import PolicyRAG  # noqa: E402
from taxrag.retrieve import Retriever  # noqa: E402

USAGE = __doc__


def build_rag(strategy: str, mode: str, use_rerank: bool) -> PolicyRAG:
    chunks = read_jsonl(PROCESSED_DIR / f"chunks_{strategy}.jsonl")
    if not chunks:
        raise SystemExit(f"缺少切分结果，请先运行 02_build_index.py {strategy}")
    retriever = Retriever(chunks, strategy, EMBED_MODEL, mode=mode, use_rerank=use_rerank)
    return PolicyRAG(retriever)


def show(rag: PolicyRAG, question: str) -> None:
    r = rag.answer(question)
    print("\n" + "=" * 72)
    print(f"问：{question}")
    print("-" * 72)
    print(f"答：{r['answer']}")
    if r["citations"]:
        print("-" * 72)
        print("引用来源：")
        for c in r["citations"]:
            valid = f"（执行至 {c['valid_to']}）" if c["valid_to"] else ""
            print(f"  [{c['n']}] {c['title']}{valid}")
            print(f"      {c['doc_no'] or c['doc_id']}  重排分 {c['score']}")
    else:
        print("（未检索到可支撑回答的政策片段，已拒答）")
    print("=" * 72)


def main() -> None:
    argv = sys.argv[1:]
    strategy, mode, use_rerank = "article", "hybrid", True

    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-s":
            strategy = argv[i + 1]; i += 2
        elif a == "-m":
            mode = argv[i + 1]; i += 2
        elif a == "--no-rerank":
            use_rerank = False; i += 1
        elif a in ("-h", "--help"):
            print(USAGE); return
        else:
            positional.append(a); i += 1

    rag = build_rag(strategy, mode, use_rerank)
    print(f"配置：切分={strategy} 检索={mode} 重排={'开' if use_rerank else '关'}")

    if positional:
        show(rag, " ".join(positional))
        return

    print("输入问题回车提问，直接回车退出。")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        show(rag, q)


if __name__ == "__main__":
    main()
