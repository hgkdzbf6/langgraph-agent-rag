"""端到端评测：对比不同配置下的延迟、token 消耗、答案质量。"""
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
class ConfigResult:
    config_name: str
    total_latency_s: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    n_queries: int = 0
    avg_latency_s: float = 0.0
    avg_tokens: float = 0.0


@dataclass
class E2EMetrics:
    results: list[ConfigResult] = field(default_factory=list)


class E2EEvaluator:
    def __init__(self, llm_factory, dataset_path: str | None = None):
        """
        llm_factory: callable(config) -> LLMClient，每次评测创建新客户端。
        """
        self.llm_factory = llm_factory
        if dataset_path is None:
            dataset_path = str(Path(__file__).parent / "dataset.json")
        with open(dataset_path, encoding="utf-8") as f:
            self.dataset = json.load(f)

    def evaluate(self, configs: dict[str, dict] | None = None,
                 max_queries: int = 3) -> E2EMetrics:
        """对多组配置运行评测。

        configs: {"配置名": {"enable_reflection": bool, "enable_complexity_check": bool, ...}}
        """
        if configs is None:
            configs = {
                "full": {"enable_reflection": True, "enable_complexity_check": True},
                "no_reflection": {"enable_reflection": False, "enable_complexity_check": True},
                "no_planner": {"enable_reflection": True, "enable_complexity_check": False},
                "minimal": {"enable_reflection": False, "enable_complexity_check": False},
            }

        metrics = E2EMetrics()
        queries = [item["query"] for item in self.dataset[:max_queries]]

        for cfg_name, overrides in configs.items():
            cfg = Config()
            for k, v in overrides.items():
                setattr(cfg, k, v)

            tracer = Tracer()
            cost = CostTracker(model=cfg.llm.model)
            llm = self.llm_factory(cfg, tracer, cost)
            graph = build_graph(llm, cfg, tracer)

            total_latency = 0.0
            for q in queries:
                t0 = time.time()
                graph.invoke({"question": q}, config={"recursion_limit": 120})
                total_latency += time.time() - t0

            result = ConfigResult(
                config_name=cfg_name,
                total_latency_s=total_latency,
                total_tokens=cost.total.total_tokens,
                total_cost=cost.total.cost_cny,
                n_queries=len(queries),
                avg_latency_s=total_latency / max(len(queries), 1),
                avg_tokens=cost.total.total_tokens / max(len(queries), 1),
            )
            metrics.results.append(result)

        return metrics

    def report(self) -> str:
        metrics = self.evaluate()
        lines = ["==== 端到端评测 ===="]
        lines.append(f"  {'配置':<16} {'平均延迟':>8} {'平均Token':>10} {'总成本':>12}")
        lines.append("  " + "-" * 50)
        for r in metrics.results:
            lines.append(
                f"  {r.config_name:<16} {r.avg_latency_s:>7.1f}s "
                f"{r.avg_tokens:>9.0f} ¥{r.total_cost:>10.6f}"
            )
        return "\n".join(lines)
