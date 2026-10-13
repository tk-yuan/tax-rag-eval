"""JSONL 读写与进度显示的小工具。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class Progress:
    """轻量进度条，避免为了一个进度显示引入 tqdm 依赖。"""

    def __init__(self, total: int, label: str = "", every: int = 20) -> None:
        self.total = total
        self.label = label
        self.every = every
        self.start = time.time()

    def tick(self, i: int) -> None:
        if i % self.every and i != self.total:
            return
        pct = i / max(self.total, 1)
        bar = "#" * int(pct * 30)
        elapsed = time.time() - self.start
        sys.stdout.write(f"\r  {self.label} [{bar:<30}] {i}/{self.total} {elapsed:5.1f}s")
        sys.stdout.flush()

    def done(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()
