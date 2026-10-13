"""切分策略：同一份语料，三种切法，供消融实验对照。

为什么切分值得单列一章：政策文本有它自己的结构——以「一、」「二、」分条，
部分以「第X条」编号。通用切分（固定长度 / 递归字符）会把一条政策拦腰截断，
检索到的片段语义不完整，模型只能看到「半句话」，自然答不准。

三种策略：
  fixed     —— 固定窗口。最简单，作为下界基线。
  recursive —— 递归按段落/句子逐级切（LangChain 默认思路）。通用做法。
  article   —— 按条款边界切。领域感知做法，本项目主张的「正确姿势」。

消融实验要回答的是：领域感知切分到底值多少分。
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 条款边界：「一、」「二、」… 或 「第X条」
ARTICLE_SPLIT = re.compile(
    r"(?m)(?=^\s*(?:第[一二三四五六七八九十百零]+\s*条|[一二三四五六七八九十]+\s*、))"
)

# 单块上限：条款切分后仍可能遇到超长段落，超过则再递归切一次
MAX_CHUNK_CHARS = 800

STRATEGIES = ("fixed", "recursive", "article")


def chunk_fixed(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """固定窗口切分：每 size 字一刀，相邻保留 overlap 字重叠。"""
    text = text.strip()
    if not text:
        return []
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + size].strip())
        if i + size >= len(text):
            break
        i += size - overlap
    return [c for c in out if c]


def chunk_recursive(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    """递归字符切分：优先在段落、换行、句号处断开。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "；", "，", ""],
        keep_separator=True,
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def chunk_by_article(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """按政策条款边界切分，保证每个片段是完整的「一条」。

    先按条款切；仍有超长段落时，用递归切分兜底（不跨条款）。
    """
    text = text.strip()
    if not text:
        return []

    out: list[str] = []
    for part in ARTICLE_SPLIT.split(text):
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            out.append(part)
        else:
            out.extend(chunk_recursive(part, size=max_chars, overlap=0))
    return out


def chunk(text: str, strategy: str, **kw) -> list[str]:
    """统一入口。"""
    if strategy == "fixed":
        return chunk_fixed(text, **kw)
    if strategy == "recursive":
        return chunk_recursive(text, **kw)
    if strategy == "article":
        return chunk_by_article(text, **kw)
    raise ValueError(f"未知切分策略：{strategy}（可选 {STRATEGIES}）")
