"""FAISS 向量库封装。

IndexFlatIP + L2 归一化向量 == cosine 相似度。
支持 build / add / search / save / load，metadata 单独存 JSON。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .embedder import Embedder

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.index = None       # faiss.Index
        self.meta: list[dict[str, Any]] = []

    def _ensure_index(self) -> None:
        if self.index is None:
            if faiss is None:
                raise RuntimeError("未安装 faiss，请 pip install faiss-cpu")
            self.index = faiss.IndexFlatIP(self.embedder.dim)

    def add(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        vecs = self.embedder.embed([c["text"] for c in chunks])
        self._ensure_index()
        self.index.add(np.ascontiguousarray(vecs))
        self.meta.extend(chunks)

    def build(self, chunks: list[dict[str, Any]]) -> None:
        self.index = None
        self.meta = []
        self.add(chunks)

    def search(self, query: str, topk: int = 20) -> list[tuple[dict[str, Any], float]]:
        if self.index is None or len(self.meta) == 0:
            return []
        qv = self.embedder.embed([query])
        self._ensure_index()
        k = min(topk, len(self.meta))
        scores, ids = self.index.search(np.ascontiguousarray(qv), k)
        return [(self.meta[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    def save(self, index_path: str, meta_path: str) -> None:
        if faiss is not None and self.index is not None:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, index_path)
        Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
        Path(meta_path).write_text(json.dumps(self.meta, ensure_ascii=False))

    def load(self, index_path: str, meta_path: str) -> bool:
        if faiss is None:
            return False
        ip, mp = Path(index_path), Path(meta_path)
        if not (ip.exists() and mp.exists()):
            return False
        self.index = faiss.read_index(str(ip))
        self.meta = json.loads(mp.read_text())
        return True

    def __len__(self) -> int:
        return len(self.meta)
