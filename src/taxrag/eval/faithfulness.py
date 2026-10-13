"""答案忠实度评测（RAG Triad 中的 faithfulness）。

为什么要单独评测生成层：检索对了，不代表答得对。模型可能召回了正确的政策，
却把数字说错、或把两个条款的条件混在一起。只看检索指标会漏掉这类错误。

做法：让另一个 LLM 当裁判，把「答案」与「被引用的片段」对照，判断每句结论
是否有片段支撑。这是 RAGAS 那类框架的简化版。

**局限必须说清楚**：LLM 裁判本身有噪声，不能当 ground truth。
所以本模块只做「抽样 + 人工复核」的辅助，不作为唯一结论。
"""

from __future__ import annotations

from ..llm import DeepSeekChat

JUDGE_SYSTEM = "你是严格的评测员，只依据给定材料判断，不引入外部知识。你只输出 JSON。"

JUDGE_PROMPT = """材料（检索到的政策片段）：

{context}

待判定答案：

{answer}

请判断：答案中的每一条结论，是否都能在上面的材料中找到依据？

输出 JSON：
{{
  "supported": true/false,
  "score": 0.0-1.0,
  "unsupported_claims": ["答案中找不到依据的结论，逐条列出"]
}}

score 的含义：有依据的结论数 / 结论总数。若答案本就说「依据不足」，则 score 为 1.0。"""


def judge_faithfulness(question: str, answer: str, hits: list[dict], client: DeepSeekChat) -> dict:
    """返回 {'supported': bool, 'score': float, 'unsupported_claims': [...]}"""
    context = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
    try:
        out = client.complete_json(
            JUDGE_PROMPT.format(context=context, answer=answer), system=JUDGE_SYSTEM
        )
    except Exception as e:  # noqa: BLE001
        return {"supported": False, "score": 0.0, "unsupported_claims": [f"评测失败：{e}"], "error": True}
    return {
        "supported": bool(out.get("supported", False)),
        "score": float(out.get("score", 0.0)),
        "unsupported_claims": out.get("unsupported_claims", []) or [],
    }
