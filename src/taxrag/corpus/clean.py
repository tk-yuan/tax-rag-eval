"""政策原文清洗：剥掉网页模板噪音，抽出可检索的正文与元数据。

原始语料是从财政部 / 税务总局网站抓下来的公告页，499 篇共用同一套模板：

    <标题>
    财政部微信 / 财政部视频号 / 官方抖音号 / 移动客户端下载
    税政司 / 搜索 / 高级搜索 / 返回主站
    当前位置：首页>政策发布
    <标题>                      ← 正文从这里开始
    2023年08月23日来源：xxx      ← 发布日期
    财政部 税务总局公告2023年第25号  ← 文号
    <正文条款>
    ...
    特此公告。
    财政部 税务总局              ← 落款
    2023年8月18日
    附件下载：
    { var handel=document.getElementById(...) ... }   ← JS，到此为止

清洗做三件事：
  1) 定位正文区间：模板标记之前全丢，JS / 附件之前截断
  2) 抽元数据：标题、文号、发布日期、发文机关、执行截止日
  3) 输出结构化记录，供切分与检索使用

「执行截止日」值得单说：政策有效期是本研究关心的重点（过期政策不能再用），
所以清洗阶段就把「本公告执行至 2027 年 12 月 31 日」这类句子抽成结构化字段。
"""

from __future__ import annotations

import re
from pathlib import Path

# 正文起点标记：该行之后的那一行才是真标题
CONTENT_START_MARK = "当前位置"

# 正文终点标记（出现即截断）
END_MARKS = ("附件下载", "相关文章", "扫一扫", "分享到", "责任编辑")

# 命中即判定为 JS / 页面脚本残留
JS_HINTS = (
    "var ",
    "var=",
    "document.",
    "getElementById",
    "style.display",
    "innerText",
    "function(",
    "window.",
    "appendChild",
)

# 导航条固定文案（整行等于这些值就丢弃）
NAV_LINES = {
    "财政部微信",
    "财政部视频号",
    "财政部",
    "官方抖音号",
    "移动客户端Android下载",
    "移动客户端iphone下载",
    "税政司",
    "搜索",
    "高级搜索-->",
    "高级搜索",
    "返回主站",
    "中华人民共和国财政部",
    "-->",
    "--&gt;",
    "{",
    "}",
}

# 文号：从最具体到最宽松
DOC_NO_PATTERNS = (
    r"财政部\s*税务总局\s*中国证监会\s*公告\s*\d{4}\s*年第\s*\d+\s*号",
    r"财政部\s*税务总局\s*公告\s*\d{4}\s*年第\s*\d+\s*号",
    r"国家税务总局\s*公告\s*\d{4}\s*年第\s*\d+\s*号",
    r"财税\s*〔\s*\d{4}\s*〕\s*\d+\s*号",
    r"财关税\s*〔\s*\d{4}\s*〕\s*\d+\s*号",
    r"第\s*\d{4}\s*年第\s*\d+\s*号",
    # 兜底：多部门联合发文（如 财政部 海关总署 税务总局公告2024年第X号）
    r"[一-龥]{2,8}(?:\s+[一-龥]{2,8}){0,3}\s*公告\s*\d{4}\s*年第\s*\d+\s*号",
)

ORG_NAMES = ("财政部", "国家税务总局", "税务总局", "中国证监会", "国务院", "海关总署")

_DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")


def _fmt_date(m: re.Match) -> str:
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"


def split_body(raw: str) -> list[str]:
    """切出正文行：跳过模板导航，遇 JS / 附件即停。"""
    lines = [ln.strip() for ln in raw.replace("\r\n", "\n").split("\n")]

    start = 0
    for i, ln in enumerate(lines):
        if CONTENT_START_MARK in ln:
            start = i + 1

    body: list[str] = []
    for ln in lines[start:]:
        if not ln:
            continue
        if any(ln.startswith(m) for m in END_MARKS):
            break
        if any(h in ln for h in JS_HINTS):
            break
        if ln in NAV_LINES:
            continue
        body.append(ln)
    return body


def extract_meta(body: list[str], fallback_title: str) -> dict:
    """从正文行里抽标题、文号、发布日期、发文机关、执行截止日。"""
    text = "\n".join(body)
    head = body[:8]

    title = head[0] if head else fallback_title
    title = re.sub(r"^[《\s]+|[》\s]+$", "", title)

    doc_no = ""
    for ln in body[:12]:
        for pat in DOC_NO_PATTERNS:
            m = re.search(pat, ln)
            if m:
                doc_no = re.sub(r"\s+", "", m.group(0))
                break
        if doc_no:
            break

    issued = ""
    for ln in head:
        m = _DATE_RE.search(ln)
        if m:
            issued = _fmt_date(m)
            break

    issuer = ""
    for ln in body[1:12]:
        if len(ln) <= 40 and any(o in ln for o in ORG_NAMES):
            issuer = ln
            break
    if issuer:
        # 发文机关行常与文号同处一行，把文号部分剥掉只留机关
        issuer = re.sub(r"公告\s*\d{4}\s*年第\s*\d+\s*号.*$", "", issuer)
        issuer = re.sub(r"〔\s*\d{4}\s*〕.*$", "", issuer)
        issuer = issuer.strip(" ，,。、：:")

    valid_to = ""
    m = re.search(r"执行至\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if not m:
        m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*止", text)
    if m:
        valid_to = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    return {
        "title": title,
        "doc_no": doc_no,
        "issued": issued,
        "issuer": issuer,
        "valid_to": valid_to,
    }


def clean_document(path: Path, source_dir: str = "raw_policies") -> dict:
    """清洗单篇政策，返回结构化记录。"""
    raw = path.read_text(encoding="utf-8", errors="replace")
    body = split_body(raw)
    meta = extract_meta(body, fallback_title=path.stem)
    text = "\n".join(body)

    return {
        "doc_id": path.stem,
        "source_file": f"{source_dir}/{path.name}",
        "raw_chars": len(raw),
        "text": text,
        "chars": len(text),
        "kept_ratio": round(len(text) / max(len(raw), 1), 3),
        **meta,
    }
