"""Token 成本聚合：从 LLM 响应 usage 字段累计，按 node/tool/run 维度统计。

智谱 GLM usage 形如：
  {"prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M,
   "prompt_tokens_details": {...}, "completion_tokens_details": {...}}
不同模型单价不同，价格表可按需调整。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

# 单价表（元 / 1K tokens）。仅作示意，按网关实际计费调整。
PRICE_CNY_PER_1K = {
    "GLM-5.2":  {"input": 0.005, "output": 0.015},
    "GLM-4.5":  {"input": 0.005, "output": 0.015},
    "default":  {"input": 0.005, "output": 0.015},
}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_cny: float = 0.0


@dataclass
class CostTracker:
    model: str = "GLM-5.2"
    by_scope: dict[str, Usage] = field(default_factory=lambda: defaultdict(Usage))
    total: Usage = field(default_factory=Usage)

    def add(self, usage: dict | None, scope: str = "unscoped", model: str | None = None) -> None:
        """记录一次 LLM 调用的 usage。scope 可为节点名/工具名，便于归因。"""
        if not usage:
            return
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        tt = usage.get("total_tokens", pt + ct)
        price = PRICE_CNY_PER_1K.get(model or self.model, PRICE_CNY_PER_1K["default"])
        cost = pt / 1000 * price["input"] + ct / 1000 * price["output"]

        u = self.by_scope[scope]
        u.prompt_tokens += pt
        u.completion_tokens += ct
        u.total_tokens += tt
        u.cost_cny += cost

        self.total.prompt_tokens += pt
        self.total.completion_tokens += ct
        self.total.total_tokens += tt
        self.total.cost_cny += cost

    def report(self) -> str:
        lines = ["==== Token & Cost Report ===="]
        for scope, u in sorted(self.by_scope.items()):
            lines.append(
                f"  [{scope}] in={u.prompt_tokens} out={u.completion_tokens} "
                f"total={u.total_tokens}  ¥{u.cost_cny:.6f}"
            )
        t = self.total
        lines.append("  " + "-" * 40)
        lines.append(
            f"  [TOTAL] in={t.prompt_tokens} out={t.completion_tokens} "
            f"total={t.total_tokens}  ¥{t.cost_cny:.6f}"
        )
        return "\n".join(lines)
