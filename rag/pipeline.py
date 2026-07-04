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


_SKIP_DIRS = {".git", ".obsidian", ".mimocode", "__pycache__", ".embed_cache",
              "node_modules", ".venv", "venv"}
_SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp",
                  ".pdf", ".zip", ".tar", ".gz", ".canvas", ".excalidraw"}


def _load_docs_from_dir(dir_: str) -> list[dict[str, str]]:
    """递归读取目录下 .txt/.md 文件作为文档。"""
    docs: list[dict[str, str]] = []
    p = Path(dir_)
    if not p.exists():
        return docs
    for f in sorted(p.rglob("*")):
        # 跳过指定目录
        if any(part in _SKIP_DIRS for part in f.parts):
            continue
        # 跳过非文本后缀
        if f.suffix.lower() in _SKIP_SUFFIXES:
            continue
        if f.suffix.lower() in (".txt", ".md"):
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    docs.append({"text": text,
                                 "source": str(f.relative_to(p))})
            except Exception:
                continue
    return docs


def _is_obsidian_vault(dir_: str) -> bool:
    """检测是否为 Obsidian 仓库：含 .obsidian 配置目录。"""
    return (Path(dir_) / ".obsidian").is_dir()


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
        """通用目录索引：自动识别 Obsidian 仓库。

        若检测到 Obsidian 特征（.obsidian 目录或大量 .md），走 Obsidian 专用
        清洗 + 中文结构化切分；否则回退到通用加载。
        """
        if _is_obsidian_vault(dir_):
            return self.ingest_obsidian(dir_)
        return self.ingest(_load_docs_from_dir(dir_))

    def ingest_obsidian(self, dir_: str) -> int:
        """Obsidian 仓库索引：清洗 + 中文结构化切分。"""
        from .obsidian_loader import load_obsidian_notes
        from .cn_chunking import chunk_notes
        with (self.tracer.span("rag:ingest_obsidian", dir=dir_) if self.tracer else _noop_ctx()):
            notes = load_obsidian_notes(dir_)
            chunks = chunk_notes(notes, self.cfg.chunk_size, self.cfg.chunk_overlap)
            self.store.build(chunks)
            self.store.save(self.cfg.index_path, self.cfg.meta_path)
            return len(chunks)

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
        out: list[str] = []
        for c in reranked:
            ck = c["chunk"]
            source = ck.get("source", "?")
            title = ck.get("title", "")
            heading = ck.get("heading_path", "")
            idx = ck.get("index", 0)
            tags = ck.get("tags", [])
            header = f"[{source}#{idx}]"
            if title:
                header += f" 《{title}》"
            if heading and heading != title:
                header += f" § {heading}"
            if tags:
                header += f" {tags}"
            out.append(f"{header} (score={c['score']:.3f})\n{ck['text']}")
        return out


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
