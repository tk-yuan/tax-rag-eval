"""全局配置：路径、模型、检索参数。

所有可调项集中在这里，方便做消融实验时统一改。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# 密钥只从项目根的 .env 读取（参考 .env.example）
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_policies"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = ROOT / "reports"
CACHE_DIR = ROOT / ".cache"

CORPUS_JSONL = PROCESSED_DIR / "corpus.jsonl"
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"
EVALSET_JSONL = PROCESSED_DIR / "evalset.jsonl"
RUNS_DIR = PROCESSED_DIR / "runs"

# —— 模型服务（硅基流动提供 embedding / rerank，DeepSeek 提供生成）——
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = 1024
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# —— 检索默认参数 ——
TOP_K_RECALL = 20      # 一阶段召回条数（广撒网）
TOP_K_RERANK = 3       # 二阶段精排后保留条数（精准命中）
RRF_K = 60             # 混合检索 RRF 融合常数

for _d in (PROCESSED_DIR, REPORT_DIR, CACHE_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
