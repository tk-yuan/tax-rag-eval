"""生成模型客户端（DeepSeek），带用量统计。

评测报告里要报 token 消耗与耗时——「只报 Recall 不报 cost」是常见的减分项，
所以客户端在这里就把用量累计下来，供上层汇总。
"""

from __future__ import annotations

import json
import re
import time

import httpx

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL

TIMEOUT = 120.0
MAX_RETRY = 3


class DeepSeekChat:
    def __init__(self, model: str = LLM_MODEL) -> None:
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.elapsed = 0.0

    @property
    def usage(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "elapsed_sec": round(self.elapsed, 2),
        }

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        url = DEEPSEEK_BASE_URL.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last: Exception | None = None
        for attempt in range(MAX_RETRY):
            try:
                t0 = time.time()
                r = httpx.post(url, headers=headers, json=payload, timeout=TIMEOUT)
                self.elapsed += time.time() - t0
                r.raise_for_status()
                data = r.json()
                u = data.get("usage") or {}
                self.prompt_tokens += u.get("prompt_tokens", 0)
                self.completion_tokens += u.get("completion_tokens", 0)
                self.calls += 1
                return data["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.5 ** attempt)
        raise RuntimeError(f"LLM 调用失败：{last}")

    def complete_json(self, prompt: str, system: str | None = None, max_tokens: int = 2048):
        """要求模型返回 JSON 并解析（容忍 ```json 围栏与前后杂字）。"""
        raw = self.complete(prompt, system=system, max_tokens=max_tokens)
        return parse_json(raw)


def parse_json(raw: str):
    """从模型输出里抠出 JSON 对象或数组。"""
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for pat in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pat, raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    raise ValueError(f"无法解析 JSON：{raw[:200]}")
