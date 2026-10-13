"""评测集构建：合成「问题 → 金标政策」对。

为什么这样做是对的，而不是手工标注：
  金标是**构造出来的**，不是猜的——问题由某份政策生成，那么该政策天然就是正确答案。
  这消除了标注噪声，规模也能上去。代价是问题分布偏「可直接查到」，
  真实用户会问更含糊、需要跨文件综合的问题，所以另建一个「实案子集」作对照。

关键的防作弊约束：生成问题时**禁止照抄政策标题、文号和独特原句**。
否则问题里直接含「小型微利企业」五个字，纯关键词检索就能命中，测不出语义能力。
"""

from __future__ import annotations

import random

from ..llm import DeepSeekChat

SYSTEM = (
    "你是一名中国企业的财务税务经理，擅长把税收政策条文转写成真实业务场景下的提问。"
    "你只输出 JSON，不输出任何解释。"
)

PROMPT = """下面是一份中国税收政策文件的正文。

--- 政策正文开始 ---
{text}
--- 政策正文结束 ---

请生成 {n} 个企业财务/税务人员会**真实提出**的问题，要求：

1. 问题必须**仅凭上面这份文件就能回答**（答案确实在这份文件里）。
2. 用**业务场景语言**提问，例如「我们公司打算买一台 400 万的设备，能不能当年一次扣完？」
3. **严禁**在问题里出现：政策标题、文号（如"2026年第18号"）、以及文件中独有的原句。
   也不要用"根据某公告"这类话。要让问题看起来像是**不知道答案的人**问出来的。
4. 为每个问题同时给出**参考答案**（依据文件内容，20-80 字）。

只输出如下 JSON 数组，不要有其他文字：
[{{"question": "...", "answer": "...", "difficulty": "single_hop"}}]"""


def generate_for_doc(doc: dict, client: DeepSeekChat, n: int = 2) -> list[dict]:
    """对单份政策生成评测问题。"""
    text = doc["text"]
    if len(text) > 6000:  # 超长文件截断，避免 prompt 过大
        text = text[:6000]
    try:
        items = client.complete_json(PROMPT.format(text=text, n=n), system=SYSTEM)
    except Exception:  # noqa: BLE001 —— 单条失败不影响整体
        return []
    if not isinstance(items, list):
        return []

    out = []
    for it in items:
        q = (it.get("question") or "").strip()
        if len(q) < 8:
            continue
        out.append(
            {
                "query": q,
                "answer": (it.get("answer") or "").strip(),
                "difficulty": it.get("difficulty", "single_hop"),
                "gold_doc_id": doc["doc_id"],
                "gold_doc_no": doc.get("doc_no", ""),
                "gold_title": doc.get("title", ""),
                "gold_issued": doc.get("issued", ""),
                "gold_valid_to": doc.get("valid_to", ""),
                "source": "synthetic",
            }
        )
    return out


def sample_docs(
    docs: list[dict], n: int, seed: int = 20261013, min_chars: int = 300
) -> list[dict]:
    """按年份分层抽样，保证评测集覆盖不同时期的政策。"""
    pool = [d for d in docs if len(d.get("text", "")) >= min_chars]
    rng = random.Random(seed)

    by_year: dict[str, list[dict]] = {}
    for d in pool:
        by_year.setdefault((d.get("issued") or "????")[:4], []).append(d)

    for v in by_year.values():
        rng.shuffle(v)

    picked: list[dict] = []
    years = sorted(by_year)
    i = 0
    while len(picked) < n and any(by_year.values()):
        y = years[i % len(years)]
        if by_year[y]:
            picked.append(by_year[y].pop())
        i += 1
    return picked[:n]
