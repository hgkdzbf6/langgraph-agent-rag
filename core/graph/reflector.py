"""Reflection 节点：自我纠错。

在 react_observe 判定子任务「完成」后，让 LLM 复核结果是否可靠、是否需要重做。
- review：返回 ok / retry + 反馈；
- retry：把反馈注入子任务消息，回到 react_reason 重新推理；
- 超过 max_reflections 次则强制接受当前结果。
"""
from __future__ import annotations

import json
import re

from config import Config
from observability import Tracer
from core.llm import LLMClient
from .state import AgentState, SubTask

REVIEW_PROMPT = """你是一个严格的审查者。请判断以下子任务结果是否正确、充分、可信。

子任务目标：{goal}
当前结果：{result}

仅输出 JSON：
{{ "verdict": "ok" | "retry", "feedback": "若 retry，给出具体修正建议；否则空字符串" }}
"""


def _review(llm: LLMClient, sub: SubTask, tracer: Tracer) -> tuple[str, str]:
    res = llm.chat(
        [{"role": "user",
          "content": REVIEW_PROMPT.format(goal=sub["goal"], result=sub["result"][:2000])}],
        scope="reflect",
    )
    try:
        m = re.search(r"\{.*\}", res.content, re.S)
        data = json.loads(m.group(0) if m else res.content)
        return data.get("verdict", "ok"), data.get("feedback", "")
    except Exception:
        return "ok", ""


def make_reflector(llm: LLMClient, cfg: Config, tracer: Tracer):
    def node(state: AgentState) -> AgentState:
        idx = state["current_index"]
        sub = state["subtasks"][idx]
        if sub["status"] != "done":
            return {"subtasks": state["subtasks"]}  # 未完成，不审查
        if not cfg.enable_reflection:
            trace = state.get("trace", [])
            trace.append(f"[reflector#{idx}] Reflection 已禁用，直接接受")
            return {"subtasks": state["subtasks"], "trace": trace}
        if sub["reflection_count"] >= cfg.max_reflections:
            return {"subtasks": state["subtasks"]}  # 达到上限，接受

        with tracer.span("reflector", subtask=idx, n_so_far=sub["reflection_count"]):
            verdict, feedback = _review(llm, sub, tracer)

        trace = state.get("trace", [])
        if verdict == "retry" and feedback:
            sub["reflection_count"] += 1
            sub["reflections"].append(feedback)
            sub["status"] = "running"
            sub["react_steps"] = 0
            # 注入反思，强制重新推理
            sub["messages"].append({"role": "user",
                                    "content": f"上一轮结果存在不足，请基于以下反馈重做：\n{feedback}"})
            trace.append(f"[reflector#{idx}] retry #{sub['reflection_count']}: {feedback[:80]}")
            return {"subtasks": state["subtasks"], "trace": trace}
        # ok
        trace.append(f"[reflector#{idx}] 接受结果")
        return {"subtasks": state["subtasks"], "trace": trace}
    return node
