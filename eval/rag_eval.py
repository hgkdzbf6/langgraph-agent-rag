"""RAG 召回质量评测：Recall@K、MRR、Answer Accuracy。"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field

from rag.pipeline import RAGPipeline


@dataclass
class RAGMetrics:
    recall_at_k: float = 0.0      # Top-K 中包含 expected_chunk 的比例
    mrr: float = 0.0              # Mean Reciprocal Rank
    answer_similarity: float = 0.0  # 答案语义相似度（余弦）
    n_queries: int = 0


@dataclass
class QueryResult:
    query: str
    retrieved_chunks: list[str]
    expected_chunks: list[str]
    recall: float
    rr: float  # reciprocal rank


class RAGEvaluator:
    def __init__(self, pipeline: RAGPipeline, dataset_path: str | None = None):
        self.pipeline = pipeline
        if dataset_path is None:
            dataset_path = str(Path(__file__).parent / "dataset.json")
        with open(dataset_path, encoding="utf-8") as f:
            self.dataset = json.load(f)

    def evaluate(self, topk: int = 5) -> RAGMetrics:
        """对数据集中所有 query 评测 RAG 召回质量。"""
        results: list[QueryResult] = []
        for item in self.dataset:
            expected = item.get("expected_chunks", [])
            if not expected:
                continue  # 跳过无 expected_chunk 的简单问题
            chunks = self.pipeline.query(item["query"], topn=topk)
            chunk_texts = [c.split("] ", 1)[-1] if "] " in c else c for c in chunks]

            # Recall@K: 是否有 expected 关键词出现在检索结果中
            recall = self._compute_recall(chunk_texts, expected)
            rr = self._compute_reciprocal_rank(chunk_texts, expected)
            results.append(QueryResult(
                query=item["query"],
                retrieved_chunks=chunk_texts,
                expected_chunks=expected,
                recall=recall,
                rr=rr,
            ))

        if not results:
            return RAGMetrics()

        return RAGMetrics(
            recall_at_k=sum(r.recall for r in results) / len(results),
            mrr=sum(r.rr for r in results) / len(results),
            n_queries=len(results),
        )

    def _compute_recall(self, retrieved: list[str], expected: list[str]) -> float:
        """计算 Recall：expected 中有多少在 retrieved 中出现。"""
        if not expected:
            return 1.0
        retrieved_text = " ".join(retrieved).lower()
        hits = sum(1 for e in expected if e.lower() in retrieved_text)
        return hits / len(expected)

    def _compute_reciprocal_rank(self, retrieved: list[str], expected: list[str]) -> float:
        """计算 Reciprocal Rank：第一个相关结果的排名倒数。"""
        retrieved_text = " ".join(retrieved).lower()
        for i, chunk in enumerate(retrieved):
            if any(e.lower() in chunk.lower() for e in expected):
                return 1.0 / (i + 1)
        return 0.0

    def report(self) -> str:
        metrics = self.evaluate()
        lines = [
            "==== RAG 召回质量评测 ====",
            f"  查询数: {metrics.n_queries}",
            f"  Recall@K: {metrics.recall_at_k:.3f}",
            f"  MRR: {metrics.mrr:.3f}",
        ]
        return "\n".join(lines)
