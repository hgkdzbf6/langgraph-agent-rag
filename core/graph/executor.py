"""工具执行节点：并发执行 react_reason 产出的 tool_calls，结果回填 messages。"""
from __future__ import annotations

import json

from observability import Tracer
from core.tools import execute
from core.concurrency import run_parallel
from .state import AgentState


def _exec_one(call: dict, tracer: Tracer) -> dict:
    name = call["name"]
    args = call.get("arguments", {})
    res = execute(name, args, tracer=tracer, scope=f"tool:{name}")
    return {"tool_call_id": call["id"], "content": res.to_str()}


def make_executor(tracer: Tracer):
    def node(state: AgentState) -> AgentState:
        idx = state["current_index"]
        sub = state["subtasks"][idx]
        pending = sub.get("_pending_tool_calls", [])
        if not pending:
            return {"subtasks": state["subtasks"]}
        with tracer.span("tool_executor", n_calls=len(pending),
                         tools=[p["name"] for p in pending]):
            results = run_parallel(
                lambda c: _exec_one(c, tracer),
                pending,
            )
        for r in results:
            sub["messages"].append({"role": "tool", **r})
        return {"subtasks": state["subtasks"]}
    return node
