"""端到端政策问答：检索 → 组装上下文 → 生成带引用的答案。

两个刻意的设计：

1. **强制引用**。提示词要求每个结论后标 [序号]，序号指向检索到的片段。
   没有引用就无法追溯，答案再流畅也不可信——这也是后面做 faithfulness 评测的前提。

2. **低置信度拒答**。重排分低于阈值时直接回「依据不足」，而不是让模型硬答。
   RAG 最常见的翻车方式就是「检索没找到，模型自己编」；把这条堵死，
   比多召回几个片段更能提升实际可信度。
"""

from __future__ import annotations

from ..llm import DeepSeekChat

SYSTEM = (
    "你是中国企业税务政策助手。你只依据用户提供的政策片段回答，"
    "不依据片段之外的知识补充或推测。"
)

PROMPT = """下面是检索到的政策片段，每条带序号：

{context}

请回答用户问题：{question}

要求：
1. **只依据上面的片段作答**。片段里没有的信息，不要补充，不要推测。
2. 每个结论后面标注来源序号，例如「……的税率为 5% [2]」。
3. 若片段不足以回答问题，直接回答「依据不足」并说明缺什么，不要勉强作答。
4. 回答用中文，控制在 200 字以内。"""

REFUSAL = "依据不足：现有政策片段未能支持该问题的回答。"

# 重排分阈值：低于此值认为检索结果不可信，直接拒答而非硬生成
MIN_RERANK_SCORE = 0.05


def format_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        src = h.get("doc_no") or h.get("doc_id", "")
        blocks.append(f"[{i}] {h.get('title', '')}（{src}）\n{h['text']}")
    return "\n\n".join(blocks)


class PolicyRAG:
    def __init__(self, retriever, llm: DeepSeekChat | None = None, min_score: float = MIN_RERANK_SCORE):
        self.retriever = retriever
        self.llm = llm or DeepSeekChat()
        self.min_score = min_score

    def answer(self, question: str) -> dict:
        hits = self.retriever.search(question)

        # 拒答门：最高分都过低，说明库里没有能支撑回答的材料
        top_score = hits[0]["score"] if hits else 0.0
        if not hits or top_score < self.min_score:
            return {
                "question": question,
                "answer": REFUSAL,
                "refused": True,
                "top_score": top_score,
                "citations": [],
            }

        answer = self.llm.complete(
            PROMPT.format(context=format_context(hits), question=question),
            system=SYSTEM,
            max_tokens=600,
        )
        return {
            "question": question,
            "answer": answer.strip(),
            "refused": REFUSAL in answer,
            "top_score": top_score,
            "citations": [
                {
                    "n": i,
                    "doc_id": h["doc_id"],
                    "doc_no": h.get("doc_no", ""),
                    "title": h.get("title", ""),
                    "valid_to": h.get("valid_to", ""),
                    "score": round(h["score"], 4),
                }
                for i, h in enumerate(hits, 1)
            ],
        }
