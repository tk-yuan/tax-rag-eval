"""语料清洗的单元测试。

清洗规则一旦回归，下游所有检索指标都会跟着变——所以这里用一份
仿造的真实网页文本（含导航条、JS、附件）来锁定行为。
"""

from taxrag.corpus.clean import clean_document, split_body

SAMPLE = """关于某事项的公告
财政部微信
 财政部视频号
 财政部
官方抖音号
移动客户端Android下载
移动客户端iphone下载
 税政司
2017年11月21日 星期二
 搜索
 高级搜索-->
 返回主站
 当前位置：首页>政策发布
 关于某事项的公告
 2026年03月05日来源：税政司
财政部 税务总局公告2026年第18号
 一、本公告自2026年1月1日起执行。
 二、符合条件的企业可享受优惠，执行至2027年12月31日。
 特此公告。
 财政部 税务总局
 2026年3月1日
附件下载：
 {
 var handel=document.getElementById("appendix1")
 }
相关文章：
"""


def test_split_body_drops_navigation():
    body = split_body(SAMPLE)
    joined = "\n".join(body)
    for junk in ("财政部微信", "官方抖音号", "高级搜索", "返回主站", "var handel"):
        assert junk not in joined


def test_split_body_keeps_content():
    joined = "\n".join(split_body(SAMPLE))
    assert "本公告自2026年1月1日起执行" in joined
    assert "特此公告" in joined


def test_split_body_stops_at_attachment():
    joined = "\n".join(split_body(SAMPLE))
    assert "相关文章" not in joined


def test_metadata_extraction():
    body = split_body(SAMPLE)
    from taxrag.corpus.clean import extract_meta

    m = extract_meta(body, fallback_title="fallback")
    assert m["title"] == "关于某事项的公告"
    assert m["doc_no"] == "财政部税务总局公告2026年第18号"
    assert m["issued"] == "2026-03-05"
    assert m["issuer"] == "财政部 税务总局"
    assert m["valid_to"] == "2027-12-31"


def test_clean_document_keeps_ratio(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text(SAMPLE, encoding="utf-8")
    rec = clean_document(p)
    assert rec["chars"] < rec["raw_chars"]
    assert 0 < rec["kept_ratio"] < 1
