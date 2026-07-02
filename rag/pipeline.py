"""RAG Pipeline：ingest（文档→索引）+ query（检索→重排→上下文）。

串联 chunking → embedder → vector_store → reranker，
对 Agent 工具层暴露统一的 ingest / query 接口。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from config import Config, RAGConfig
from observability import Tracer
from .chunking import chunk_documents
from .embedder import Embedder, GLMEmbedder, MockEmbedder
from .vector_store import VectorStore
from .reranker import Reranker, MMRReranker, NoopReranker


def _load_docs_from_dir(dir_: str) -> list[dict[str, str]]:
    """递归读取目录下 .txt/.md 文件作为文档。"""
    docs: list[dict[str, str]] = []
    p = Path(dir_)
    if not p.exists():
        return docs
    for f in sorted(p.rglob("*")):
        if f.suffix.lower() in (".txt", ".md"):
            docs.append({"text": f.read_text(encoding="utf-8", errors="ignore"),
                         "source": str(f.relative_to(p))})
    return docs


class RAGPipeline:
    def __init__(self, embedder: Embedder, reranker: Reranker, rag_cfg: RAGConfig,
                 tracer: Tracer | None = None) -> None:
        self.cfg = rag_cfg
        self.tracer = tracer
        self.store = VectorStore(embedder)
        self.reranker = reranker

    def ingest(self, docs: list[dict[str, str]]) -> int:
        """docs: [{"text":..., "source":...}, ...]；返回 chunk 数。"""
        with (self.tracer.span("rag:ingest", n_docs=len(docs)) if self.tracer else _noop_ctx()):
            chunks = chunk_documents(docs, self.cfg.chunk_size, self.cfg.chunk_overlap)
            self.store.build(chunks)
            self.store.save(self.cfg.index_path, self.cfg.meta_path)
            return len(chunks)

    def ingest_dir(self, dir_: str) -> int:
        return self.ingest(_load_docs_from_dir(dir_))

    def query(self, query: str, topn: int | None = None) -> list[str]:
        """检索 → 重排 → 返回片段文本列表。"""
        topn = topn or self.cfg.rerank_topn
        if self.tracer:
            with self.tracer.span("rag:query", query=query[:50]):
                return self._query(query, topn)
        return self._query(query, topn)

    def _query(self, query: str, topn: int) -> list[str]:
        hits = self.store.search(query, topk=self.cfg.retrieve_topk)
        if not hits:
            return []
        cands = [{"chunk": c, "score": s} for c, s in hits]
        reranked = self.reranker.rerank(query, cands, topn)
        return [
            f"[{c['chunk'].get('source','?')}#{c['chunk'].get('index',0)}] "
            f"(score={c['score']:.3f}) {c['chunk']['text']}"
            for c in reranked
        ]


class _noop_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


def build_pipeline(cfg: Config, tracer: Tracer | None = None,
                   *, mock: bool = False) -> RAGPipeline:
    """构造默认 pipeline。mock=True 用 MockEmbedder + NoopReranker，免 Key。"""
    if mock:
        embedder: Embedder = MockEmbedder(dim=256)
        reranker: Reranker = NoopReranker()
    else:
        embedder = GLMEmbedder(cfg.embed)
        reranker = MMRReranker(embedder)
    return RAGPipeline(embedder, reranker, cfg.rag, tracer)
