"""RAG 召回质量评测：完整指标体系。

指标：
- Precision@K：Top-K 结果中有多少是相关的
- Recall@K：相关结果中有多少被召回
- MRR：第一个相关结果的排名倒数
- NDCG@K：考虑位置的排序质量
- Hit Rate：至少命中 1 个相关结果的比例
- Answer Accuracy：最终答案与标准答案的关键词匹配度
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass, field

from rag.pipeline import RAGPipeline


@dataclass
class QueryMetrics:
    query_id: int
    query: str
    category: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    hit: bool
    n_retrieved: int
    n_relevant: int
    n_hits: int


@dataclass
class RAGReport:
    n_queries: int
    # 召回指标
    avg_precision: float
    avg_recall: float
    avg_mrr: float
    avg_ndcg: float
    hit_rate: float
    # 按类别
    by_category: dict[str, dict]
    # 逐条结果
    details: list[QueryMetrics]


class RAGEvaluator:
    def __init__(self, pipeline: RAGPipeline, dataset_path: str | None = None):
        self.pipeline = pipeline
        if dataset_path is None:
            dataset_path = str(Path(__file__).parent / "dataset.json")
        with open(dataset_path, encoding="utf-8") as f:
            self.dataset = json.load(f)

    def evaluate(self, topk: int = 5) -> RAGReport:
        details: list[QueryMetrics] = []

        for item in self.dataset:
            relevant = set(item.get("relevant_chunks", []))
            if not relevant:
                continue

            chunks = self.pipeline.query(item["query"], topn=topk)
            retrieved_files = [
                c.split("]")[0].split("[")[-1] for c in chunks
            ]

            hits = [f for f in retrieved_files if f in relevant]
            n_hits = len(hits)

            # Precision@K
            precision = n_hits / len(retrieved_files) if retrieved_files else 0.0
            # Recall@K
            recall = n_hits / len(relevant) if relevant else 0.0
            # MRR
            mrr = 0.0
            for i, f in enumerate(retrieved_files):
                if f in relevant:
                    mrr = 1.0 / (i + 1)
                    break
            # NDCG@K
            ndcg = self._ndcg(retrieved_files, relevant)
            # Hit Rate
            hit = n_hits > 0

            details.append(QueryMetrics(
                query_id=item["id"],
                query=item["query"],
                category=item.get("category", "unknown"),
                precision_at_k=precision,
                recall_at_k=recall,
                mrr=mrr,
                ndcg_at_k=ndcg,
                hit=hit,
                n_retrieved=len(retrieved_files),
                n_relevant=len(relevant),
                n_hits=n_hits,
            ))

        if not details:
            return RAGReport(0, 0, 0, 0, 0, 0, {}, [])

        # 汇总
        n = len(details)
        avg_p = sum(d.precision_at_k for d in details) / n
        avg_r = sum(d.recall_at_k for d in details) / n
        avg_mrr = sum(d.mrr for d in details) / n
        avg_ndcg = sum(d.ndcg_at_k for d in details) / n
        hit_rate = sum(1 for d in details if d.hit) / n

        # 按类别聚合
        by_cat: dict[str, list[QueryMetrics]] = {}
        for d in details:
            by_cat.setdefault(d.category, []).append(d)
        cat_stats = {}
        for cat, items in by_cat.items():
            cn = len(items)
            cat_stats[cat] = {
                "count": cn,
                "precision": sum(d.precision_at_k for d in items) / cn,
                "recall": sum(d.recall_at_k for d in items) / cn,
                "mrr": sum(d.mrr for d in items) / cn,
                "ndcg": sum(d.ndcg_at_k for d in items) / cn,
                "hit_rate": sum(1 for d in items if d.hit) / cn,
            }

        return RAGReport(
            n_queries=n,
            avg_precision=avg_p,
            avg_recall=avg_r,
            avg_mrr=avg_mrr,
            avg_ndcg=avg_ndcg,
            hit_rate=hit_rate,
            by_category=cat_stats,
            details=details,
        )

    def _ndcg(self, retrieved: list[str], relevant: set[str], k: int = 5) -> float:
        """计算 NDCG@K。"""
        # DCG
        dcg = 0.0
        for i, f in enumerate(retrieved[:k]):
            if f in relevant:
                dcg += 1.0 / math.log2(i + 2)  # i+2 因为 log2(1)=0
        # IDCG（理想排序）
        ideal = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    def report(self) -> str:
        r = self.evaluate()
        lines = [
            "=" * 60,
            "RAG 召回质量评测报告",
            "=" * 60,
            f"  评测查询数: {r.n_queries}",
            "",
            "--- 召回指标 ---",
            f"  Precision@5: {r.avg_precision:.3f}  (Top-5 中相关比例)",
            f"  Recall@5:    {r.avg_recall:.3f}  (相关结果被召回比例)",
            f"  MRR:         {r.avg_mrr:.3f}  (第一个相关结果排名倒数)",
            f"  NDCG@5:      {r.avg_ndcg:.3f}  (考虑位置的排序质量)",
            f"  Hit Rate:    {r.hit_rate:.3f}  (至少命中1个相关结果)",
            "",
            "--- 按类别 ---",
        ]
        for cat, s in sorted(r.by_category.items()):
            lines.append(
                f"  [{cat:15s}] n={s['count']:2d}  "
                f"P={s['precision']:.2f}  R={s['recall']:.2f}  "
                f"MRR={s['mrr']:.2f}  Hit={s['hit_rate']:.2f}"
            )

        lines.append("")
        lines.append("--- 逐条详情 ---")
        for d in r.details:
            status = "OK" if d.hit else "MISS"
            lines.append(
                f"  #{d.query_id:2d} [{status:4s}] {d.query[:30]:30s}  "
                f"P={d.precision_at_k:.2f} R={d.recall_at_k:.2f} "
                f"MRR={d.mrr:.2f} NDCG={d.ndcg_at_k:.2f}  "
                f"({d.n_hits}/{d.n_relevant})"
            )

        return "\n".join(lines)
