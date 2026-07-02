"""Agent 图测试：用 MockLLM 跑通 planner/react/reflector/aggregator，免 Key。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from observability import Tracer, CostTracker
from core.llm.base import LLMClient, ChatResult, ToolCall
from core.tools import builtin  # noqa: F401
from core.graph import build_graph


class MockLLM(LLMClient):
    """按预设脚本返回，模拟：规划 → 工具调用 → 最终答复 → 审查通过 → 汇总。"""
    def __init__(self, cost=None):
        self.calls = 0
        self.react_counts: dict[str, int] = {}
        self.cost = cost

    def _next(self, scope: str) -> ChatResult:
        self.calls += 1
        res = self._script(scope)
        if self.cost is not None and res.usage:
            self.cost.add(res.usage, scope=scope, model="GLM-5.2")
        return res

    def _script(self, scope: str) -> ChatResult:
        # 按 scope 分发
        if scope == "planner":
            return ChatResult(content='[{"goal": "检索框架能力"}, {"goal": "检索RAG优化"}]',
                              usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        if scope.startswith("react#"):
            # 第一次：调工具；第二次：给最终答复
            self.react_counts[scope] = self.react_counts.get(scope, 0) + 1
            if self.react_counts[scope] == 1:
                return ChatResult(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="python_exec",
                                         arguments={"code": "print(6*7)"})],
                    usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                )
            return ChatResult(content=f"子任务{scope}结果：具备 ReAct/Reflection 能力。",
                              tool_calls=[],
                              usage={"prompt_tokens": 25, "completion_tokens": 8, "total_tokens": 33})
        if scope == "reflect":
            return ChatResult(content='{"verdict":"ok","feedback":""}',
                              usage={"prompt_tokens": 15, "completion_tokens": 4, "total_tokens": 19})
        if scope == "aggregate":
            return ChatResult(content="综合答案：框架含 ReAct/Reflection/规划，RAG 重排优化召回。",
                              usage={"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40})
        return ChatResult(content="ok", usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

    def chat(self, messages, *, temperature=None, max_tokens=None, scope="llm"):
        return self._next(scope)

    def chat_with_tools(self, messages, tools, *, temperature=None, max_tokens=None, scope="llm"):
        return self._next(scope)


def test_graph_runs_end_to_end():
    tracer = Tracer()
    cost = CostTracker(model="GLM-5.2")
    llm = MockLLM(cost=cost)
    cfg = Config()
    graph = build_graph(llm, cfg, tracer)

    final = graph.invoke({"question": "框架能力与 RAG 优化"},
                         config={"recursion_limit": 30})
    assert final.get("done") is True
    assert final.get("final_answer")
    assert len(final.get("subtasks", [])) >= 1
    # 成本有累计
    assert cost.total.total_tokens > 0
    # trace 非空
    print("\n--- trace ---\n" + tracer.tree())
    print(cost.report())
