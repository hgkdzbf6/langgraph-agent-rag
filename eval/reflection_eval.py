"""Reflection 收益评测：对比有/无 Reflection 的答案质量与成本。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, field

from config import Config
from observability import Tracer, CostTracker
from core.llm import LLMClient
from core.graph import build_graph


@dataclass
class ReflectionMetrics:
    with_reflection_correct: int = 0
    without_reflection_correct: int = 0
    total_queries: int = 0
    reflection_extra_tokens: int = 0
    reflection_extra_cost: float = 0.0
    improvement_rate: float = 0.0  # 正确率提升百分比
    cost_benefit_ratio: float = 0.0  # 正确率提升 / token 增量


class ReflectionEvaluator:
    def __init__(self, llm: LLMClient, dataset_path: str | None = None):
        self.llm = llm
        if dataset_path is None:
            dataset_path = str(Path(__file__).parent / "dataset.json")
        with open(dataset_path, encoding="utf-8") as f:
            self.dataset = json.load(f)

    def evaluate(self, max_queries: int = 2) -> ReflectionMetrics:
        """对数据集运行两轮（有/无 Reflection），对比结果。"""
        metrics = ReflectionMetrics()
        queries = [item for item in self.dataset[:max_queries]
                   if item.get("expected_answer")]

        for item in queries:
            # 有 Reflection
            cfg_on = Config()
            cfg_on.enable_reflection = True
            cfg_on.enable_complexity_check = False  # 统一用 planner 路径
            tracer_on = Tracer()
            cost_on = CostTracker(model=cfg_on.llm.model)
            from core.llm.glm import GLMClient
            llm_on = GLMClient(cfg_on.llm, tracer_on, cost_on, cache=self.llm.cache)
            graph_on = build_graph(llm_on, cfg_on, tracer_on)
            t0 = time.time()
            state_on = graph_on.invoke({"question": item["query"]},
                                       config={"recursion_limit": 120})
            elapsed_on = time.time() - t0

            # 无 Reflection
            cfg_off = Config()
            cfg_off.enable_reflection = False
            cfg_off.enable_complexity_check = False
            tracer_off = Tracer()
            cost_off = CostTracker(model=cfg_off.llm.model)
            llm_off = GLMClient(cfg_off.llm, tracer_off, cost_off, cache=self.llm.cache)
            graph_off = build_graph(llm_off, cfg_off, tracer_off)
            t0 = time.time()
            state_off = graph_off.invoke({"question": item["query"]},
                                         config={"recursion_limit": 120})
            elapsed_off = time.time() - t0

            answer_on = state_on.get("final_answer", "")
            answer_off = state_off.get("final_answer", "")
            expected = item["expected_answer"]

            # 简单匹配：expected 关键词是否出现在答案中
            def _match(ans: str, exp: str) -> bool:
                exp_lower = exp.lower()
                return any(w in ans.lower() for w in exp_lower.split() if len(w) > 1)

            if _match(answer_on, expected):
                metrics.with_reflection_correct += 1
            if _match(answer_off, expected):
                metrics.without_reflection_correct += 1

            metrics.reflection_extra_tokens += (
                cost_on.total.total_tokens - cost_off.total.total_tokens
            )
            metrics.reflection_extra_cost += (
                cost_on.total.cost_cny - cost_off.total.cost_cny
            )
            metrics.total_queries += 1

        if metrics.total_queries > 0:
            rate_on = metrics.with_reflection_correct / metrics.total_queries
            rate_off = metrics.without_reflection_correct / metrics.total_queries
            metrics.improvement_rate = (rate_on - rate_off) * 100
            if metrics.reflection_extra_tokens > 0:
                metrics.cost_benefit_ratio = (
                    metrics.improvement_rate / metrics.reflection_extra_tokens * 1000
                )

        return metrics

    def report(self) -> str:
        metrics = self.evaluate()
        lines = [
            "==== Reflection 收益评测 ====",
            f"  查询数: {metrics.total_queries}",
            f"  有 Reflection 正确率: {metrics.with_reflection_correct}/{metrics.total_queries}",
            f"  无 Reflection 正确率: {metrics.without_reflection_correct}/{metrics.total_queries}",
            f"  正确率提升: {metrics.improvement_rate:+.1f}%",
            f"  Reflection 额外 token: {metrics.reflection_extra_tokens}",
            f"  Reflection 额外成本: ¥{metrics.reflection_extra_cost:.6f}",
            f"  收益比 (提升%/千token): {metrics.cost_benefit_ratio:.4f}",
        ]
        return "\n".join(lines)
