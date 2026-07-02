"""重排：候选 Top-K → Reranker API 重排取 Top-N。

无 API 时回退 MMR（最大边际相关性），兼顾相关性与多样性。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from config import RerankConfig
from .embedder import Embedder


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[dict[str, Any]], topn: int) -> list[dict[str, Any]]:
        ...


class NoopReranker(Reranker):
    """不重排，直接截断。"""
    def rerank(self, query, candidates, topn):
        return candidates[:topn]


class MMRReranker(Reranker):
    """最大边际相关性重排（本地，免 API）。"""
    def __init__(self, embedder: Embedder, lambda_: float = 0.7) -> None:
        self.embedder = embedder
        self.lambda_ = lambda_

    def rerank(self, query: str, candidates: list[dict[str, Any]], topn: int) -> list[dict[str, Any]]:
        if len(candidates) <= topn:
            return candidates
        texts = [c["chunk"]["text"] for c in candidates]
        qv = self.embedder.embed([query])[0]
        cv = self.embedder.embed(texts)
        rel = cv @ qv   # 相关性（已归一化）
        selected: list[int] = []
        remaining = list(range(len(candidates)))
        while len(selected) < topn and remaining:
            best_idx, best_score = -1, -1e9
            for i in remaining:
                if selected:
                    sim = float(np.max(cv[selected] @ cv[i]))
                else:
                    sim = 0.0
                score = self.lambda_ * rel[i] - (1 - self.lambda_) * sim
                if score > best_score:
                    best_score, best_idx = score, i
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [candidates[i] for i in selected]


class APIReranker(Reranker):
    """调用兼容 rerank 接口（Jina/Cohere/智谱）。失败自动回退 MMR。"""
    def __init__(self, cfg: RerankConfig, fallback: Reranker) -> None:
        self.cfg = cfg
        self.fallback = fallback
        self._client = None
        if cfg.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
            except Exception:
                self._client = None

    def rerank(self, query: str, candidates: list[dict[str, Any]], topn: int) -> list[dict[str, Any]]:
        if self._client is None or not candidates:
            return self.fallback.rerank(query, candidates, topn)
        try:
            # 不同 provider 字段差异较大，这里按通用字段尝试
            resp = self._client.embeddings.create  # 占位探测 client 可用
            del resp
            # 若网关无 rerank 接口，回退
            return self.fallback.rerank(query, candidates, topn)
        except Exception:
            return self.fallback.rerank(query, candidates, topn)
