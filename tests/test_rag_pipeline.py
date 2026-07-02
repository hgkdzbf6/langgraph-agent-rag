"""RAG pipeline 测试（用 MockEmbedder + NoopReranker，免 Key 免 faiss 也基本可跑）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RAGConfig
from observability import Tracer
from rag import MockEmbedder, NoopReranker, MMRReranker, RAGPipeline
from rag.chunking import chunk_text, chunk_documents


def test_chunking_overlap_and_size():
    text = "句子一。句子二。句子三。" * 20
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)  # overlap 放宽容差


def test_chunk_documents_metadata():
    docs = [{"text": "内容一。内容二。", "source": "a"},
            {"text": "内容三。", "source": "b"}]
    chunks = chunk_documents(docs, chunk_size=100, overlap=0)
    assert len(chunks) >= 2
    assert "source" in chunks[0]


def test_pipeline_mock_query():
    embedder = MockEmbedder(dim=64)
    rag = RAGPipeline(embedder, NoopReranker(), RAGConfig(chunk_size=100, chunk_overlap=0,
                                                          retrieve_topk=5, rerank_topn=3))
    docs = [
        {"text": "LangGraph 用于构建有状态的 Agent 工作流。", "source": "a"},
        {"text": "FAISS 是高效的向量检索库。", "source": "b"},
        {"text": "ReAct 是推理加行动的范式。", "source": "c"},
    ]
    n = rag.ingest(docs)
    assert n >= 3
    results = rag.query("Agent 工作流")
    assert isinstance(results, list)


def test_mmr_reranker_diversity():
    embedder = MockEmbedder(dim=64)
    reranker = MMRReranker(embedder, lambda_=0.5)
    cands = [{"chunk": {"text": f"doc {i}", "source": "s", "index": i}, "score": 1.0 - i * 0.1}
             for i in range(10)]
    out = reranker.rerank("query", cands, topn=3)
    assert len(out) == 3
