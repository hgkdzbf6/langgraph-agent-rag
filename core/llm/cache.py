"""LLM 响应缓存：对相同 (model, messages, tools) 的调用做本地缓存。

用 SHA256 哈希 key，支持 TTL 过期和容量限制。
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .base import ChatResult, ToolCall


def _hash_key(model: str, messages: list[dict], tools: list[dict] | None = None) -> str:
    """生成缓存 key：model + messages + tools 的 SHA256。"""
    raw = json.dumps({"model": model, "messages": messages, "tools": tools},
                     sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class LLMCache:
    def __init__(self, max_size: int = 256, ttl_seconds: int = 3600):
        self._store: dict[str, tuple[ChatResult, float]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds

    def get(self, model: str, messages: list[dict],
            tools: list[dict] | None = None) -> ChatResult | None:
        key = _hash_key(model, messages, tools)
        entry = self._store.get(key)
        if entry is None:
            return None
        result, ts = entry
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        return result

    def put(self, model: str, messages: list[dict], result: ChatResult,
            tools: list[dict] | None = None) -> None:
        if len(self._store) >= self._max_size:
            # 简单 FIFO 淘汰：删最早插入的条目
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        key = _hash_key(model, messages, tools)
        self._store[key] = (result, time.time())

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)
