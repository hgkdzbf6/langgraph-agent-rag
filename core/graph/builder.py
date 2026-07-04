"""组装 LangGraph StateGraph。

拓扑：
  START → planner
  planner → react_reason（开始第一个子任务）

  每个子任务循环：
      react_reason → tool_executor(并发) → react_observe
      react_observe 之后路由：
        - 子任务给出最终答复(status=done) → reflector
        - 否则（仍需工具）且未超步数 → react_reason
        - 超步数 → 强制完成 → reflector
      reflector 复核：
        - retry → react_reason（带反馈，重新推理）
        - ok   → advance
      advance：推进 current_index 到下一子任务并置 running
      advance 路由：
        - 还有子任务 → react_reason
        - 全部完成   → aggregator → END

注意：所有状态变更必须通过节点返回 dict 体现；
条件函数（add_conditional_edges 的 router）只读不写。
"""
from __future__ import annotations

from config import Config
from observability import Tracer
from core.llm import LLMClient
from .state import AgentState
from .planner import make_planner
from .complexity import make_complexity_check, make_simple_subtask
from .react import make_react_reason, make_react_observe
from .executor import make_executor
from .reflector import make_reflector


def _current_sub(state: AgentState):
    idx = state.get("current_index", 0)
    subs = state.get("subtasks", [])
    return idx, (subs[idx] if idx < len(subs) else None)


def make_advance(cfg: Config):
    """推进到下一子任务：current_index += 1（可能越界，由 after_advance 判定），
    并把新子任务（若有效）置 running。"""
    def node(state: AgentState) -> AgentState:
        subs = state["subtasks"]
        idx = state.get("current_index", 0)
        next_idx = idx + 1
        if next_idx < len(subs):
            subs[next_idx]["status"] = "running"
        return {"current_index": next_idx, "subtasks": subs}
    return node


def make_direct_answer(tracer: Tracer):
    """快速路径：单子任务直接取结果作为最终答案，跳过 Aggregator LLM 调用。"""
    def node(state: AgentState) -> AgentState:
        subs = state["subtasks"]
        result = subs[0]["result"] if subs else ""
        trace = state.get("trace", [])
        trace.append("[direct_answer] 单子任务直接返回结果")
        return {"final_answer": result, "done": True, "trace": trace}
    return node


def make_aggregator(llm: LLMClient, tracer: Tracer):
    def node(state: AgentState) -> AgentState:
        subs = state["subtasks"]
        with tracer.span("aggregator", n_subtasks=len(subs)):
            # 截断每个子任务结果，防止总输入过长
            MAX_RESULT_LEN = 1500
            summary = "\n\n".join(
                f"### 子任务{i+1}: {s['goal']}\n结果: {s['result'][:MAX_RESULT_LEN]}"
                for i, s in enumerate(subs)
            )
            res = llm.chat(
                [{"role": "user",
                  "content": (f"原始问题：{state['question']}\n\n"
                              f"以下是各子任务的结果：\n{summary}\n\n"
                              f"请综合这些结果，给出对原始问题的完整最终答案。")}],
                scope="aggregate",
            )
        trace = state.get("trace", [])
        trace.append("[aggregator] 生成最终答案")
        return {"final_answer": res.content, "done": True, "trace": trace}
    return node


def build_graph(llm: LLMClient, cfg: Config, tracer: Tracer):
    from langgraph.graph import StateGraph, START, END

    g = StateGraph(AgentState)
    g.add_node("planner", make_planner(llm, cfg, tracer))
    g.add_node("react_reason", make_react_reason(llm, cfg, tracer))
    g.add_node("tool_executor", make_executor(tracer))
    g.add_node("react_observe", make_react_observe(cfg, tracer, llm))
    g.add_node("reflector", make_reflector(llm, cfg, tracer))
    g.add_node("advance", make_advance(cfg))
    g.add_node("aggregator", make_aggregator(llm, tracer))
    g.add_node("direct_answer", make_direct_answer(tracer))

    if cfg.enable_complexity_check:
        g.add_node("complexity_check", make_complexity_check(llm, cfg, tracer))
        g.add_node("simple_subtask", make_simple_subtask())
        g.add_edge(START, "complexity_check")

        def after_complexity(state: AgentState) -> str:
            # simple / medium 都直接作为单任务（无需拆解），只有 hard 走 planner
            if state.get("complexity") in ("simple", "medium"):
                return "simple_subtask"
            return "planner"

        g.add_conditional_edges(
            "complexity_check", after_complexity,
            {"simple_subtask": "simple_subtask", "planner": "planner"},
        )
        g.add_edge("simple_subtask", "react_reason")
    else:
        g.add_edge(START, "planner")

    g.add_edge("planner", "react_reason")
    g.add_edge("react_reason", "tool_executor")
    g.add_edge("tool_executor", "react_observe")

    # react_observe 之后：done → reflector；否则 → react_reason（受步数限制）
    def after_observe(state: AgentState) -> str:
        idx, sub = _current_sub(state)
        if sub is None or sub["status"] == "done":
            return "reflector"
        return "react_reason"  # react_observe 节点已处理步数上限强制完成

    g.add_conditional_edges(
        "react_observe", after_observe,
        {"react_reason": "react_reason", "reflector": "reflector"},
    )

    # reflector 之后：retry → react_reason；ok → advance
    def after_reflect(state: AgentState) -> str:
        idx, sub = _current_sub(state)
        if sub is None:
            return "advance"
        if sub["status"] != "done":
            return "react_reason"   # reflection 触发了 retry
        return "advance"

    g.add_conditional_edges(
        "reflector", after_reflect,
        {"react_reason": "react_reason", "advance": "advance"},
    )

    # advance 之后：还有子任务 → react_reason；单子任务 → direct_answer（跳过 aggregator）；否则 → aggregator
    def after_advance(state: AgentState) -> str:
        idx = state.get("current_index", 0)
        if idx < len(state.get("subtasks", [])):
            return "react_reason"
        # 单子任务（无论 simple/complex）直接返回结果，跳过 Aggregator 省一次 LLM 调用
        if len(state.get("subtasks", [])) == 1:
            return "direct_answer"
        return "aggregator"

    g.add_conditional_edges(
        "advance", after_advance,
        {"react_reason": "react_reason", "aggregator": "aggregator",
         "direct_answer": "direct_answer"},
    )

    g.add_edge("aggregator", END)
    return g.compile()
