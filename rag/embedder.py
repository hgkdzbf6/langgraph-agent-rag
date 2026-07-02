"""Embedding 客户端：默认走智谱 embedding-3 兼容接口，带 hash 本地缓存。

提供 Embedder 抽象与两个实现：
- GLMEmbedder：调 API；
- MockEmbedder：确定性 hash 向量，免 Key 供测试/离线。
"""
from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from config import EmbedConfig


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """返回 [N, dim] float32，已 L2 归一化。"""
        ...


class GLMEmbedder(Embedder):
    def __init__(self, cfg: EmbedConfig) -> None:
        self.cfg = cfg
        self.dim = cfg.dim
        self._cache_path = Path(cfg.cache_dir) / "embed_cache.json"
        self._cache: dict[str, list[float]] = self._load_cache()
        self._client = None
        if cfg.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
            except Exception:
                self._client = None

    def _load_cache(self) -> dict[str, list[float]]:
        if self._cache_path.exists():
            try:
                return json.loads(self._cache_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache))

    @staticmethod
    def _key(t: str) -> str:
        return hashlib.md5(t.encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        results: list[list[float] | None] = [None] * len(texts)
        todo: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            k = self._key(t)
            if k in self._cache:
                results[i] = self._cache[k]
            else:
                todo.append((i, t))

        if todo and self._client is not None:
            # 分批，避免单次请求过大
            BATCH = 16
            for b in range(0, len(todo), BATCH):
                batch = todo[b:b + BATCH]
                resp = self._client.embeddings.create(
                    model=self.cfg.model, input=[t for _, t in batch]
                )
                for (idx, txt), item in zip(batch, resp.data):
                    vec = item.embedding
                    results[idx] = vec
                    self._cache[self._key(txt)] = vec
            self._save_cache()

        # 仍未拿到（无 client 或失败）→ 回退 mock 向量，保证不阻断
        for i, t in enumerate(texts):
            if results[i] is None:
                results[i] = _hash_vector(t, self.dim)

        arr = np.asarray(results, dtype=np.float32)
        # L2 归一化
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class MockEmbedder(Embedder):
    """确定性 hash 向量，免 Key。"""
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        arr = np.asarray([_hash_vector(t, self.dim) for t in texts], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def _hash_vector(text: str, dim: int) -> list[float]:
    """基于文本 hash 生成确定性伪向量（仅用于兜底/测试，无语义）。"""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # 扩展到 dim 长度
    buf = (h * ((dim // len(h)) + 1))[:dim]
    vec = [(b - 128) / 128.0 for b in buf]
    return vec
