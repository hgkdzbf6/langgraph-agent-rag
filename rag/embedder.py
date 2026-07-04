"""Embedding 客户端：默认走智谱 embedding-3 兼容接口，带 hash 本地缓存。

提供 Embedder 抽象与两个实现：
- GLMEmbedder：调 API；
- MockEmbedder：确定性 hash 向量，免 Key 供测试/离线。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
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
        # 双格式缓存：优先用 numpy 二进制（快），兼容旧 JSON
        self._cache_dir = Path(cfg.cache_dir)
        self._cache_npz = self._cache_dir / "embed_cache.npz"
        self._cache_json = self._cache_dir / "embed_cache.json"
        self._cache: dict[str, np.ndarray] = self._load_cache()
        self._dirty = False  # 标记是否有新缓存需落盘
        self._client = None
        if cfg.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
            except Exception:
                self._client = None

    def _load_cache(self) -> dict[str, np.ndarray]:
        """加载缓存：优先 numpy 二进制，回退旧 JSON 并自动迁移。"""
        # 1. numpy 格式（快）
        if self._cache_npz.exists():
            try:
                data = np.load(self._cache_npz, allow_pickle=False)
                keys = data["keys"]
                vecs = data["vecs"]  # [N, dim] float32
                return {str(k): vecs[i] for i, k in enumerate(keys)}
            except Exception:
                pass
        # 2. 旧 JSON 格式（慢，仅用于迁移）
        if self._cache_json.exists():
            try:
                raw = json.loads(self._cache_json.read_text())
                # 转成 numpy 并立即迁移到 npz
                migrated = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
                self._save_cache_npz(migrated)
                return migrated
            except Exception:
                pass
        return {}

    def _save_cache_npz(self, cache: dict[str, np.ndarray]) -> None:
        """以 numpy 二进制保存，反序列化比 JSON 快数十倍。"""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        if not cache:
            return
        keys = np.asarray(list(cache.keys()))
        vecs = np.stack(list(cache.values())).astype(np.float32)
        np.savez(self._cache_npz, keys=keys, vecs=vecs)

    def _save_cache(self) -> None:
        if not self._dirty:
            return
        self._save_cache_npz(self._cache)
        # 迁移成功后删除旧 JSON
        if self._cache_json.exists():
            try:
                self._cache_json.unlink()
            except Exception:
                pass
        self._dirty = False

    @staticmethod
    def _key(t: str) -> str:
        return hashlib.md5(t.encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # 用数组暂存，命中缓存直接取 numpy 向量（零拷贝）
        results: list[np.ndarray | None] = [None] * len(texts)
        todo: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            k = self._key(t)
            cached = self._cache.get(k)
            if cached is not None:
                results[i] = cached
            else:
                todo.append((i, t))

        if todo and self._client is not None:
            # 大批量 embedding（API 单次支持上百条），失败自动重试 + 降级
            BATCH = 64
            MAX_RETRIES = 3
            import concurrent.futures as _cf
            batches = [todo[b:b + BATCH] for b in range(0, len(todo), BATCH)]

            def _embed_batch(batch):
                """对单个批次调用 API，带重试。返回 [(idx, txt, vec), ...]。"""
                for attempt in range(MAX_RETRIES):
                    try:
                        resp = self._client.embeddings.create(
                            model=self.cfg.model, input=[t for _, t in batch]
                        )
                        return [(idx, txt, np.asarray(item.embedding, dtype=np.float32))
                                for (idx, txt), item in zip(batch, resp.data)]
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(0.5 * (attempt + 1))
                        else:
                            return [("__error__", str(e), None)]  # 整批失败
                return []

            # 并发请求多个批次（embedding API 通常允许并发）
            n_workers = min(8, len(batches))
            with _cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
                for batch_results in ex.map(_embed_batch, batches):
                    for idx, txt, vec in batch_results:
                        if idx == "__error__":
                            continue  # 失败的 chunk 留给后面的 mock 兜底
                        results[idx] = vec
                        self._cache[self._key(txt)] = vec
                        self._dirty = True
            self._save_cache()

        # 仍未拿到（无 client 或失败）→ 回退 mock 向量，保证不阻断
        for i, t in enumerate(texts):
            if results[i] is None:
                results[i] = np.asarray(_hash_vector(t, self.dim), dtype=np.float32)

        # stack 成矩阵（results 元素已是 ndarray）
        arr = np.stack([np.asarray(r, dtype=np.float32) for r in results])
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
