"""复杂度前置判断节点：区分简单/复杂问题，避免简单问题走完整 Planner 流程。

简单问题直接创建单个 SubTask 进入 ReAct，跳过 Planner + Aggregator。
"""
from __future__ import annotations

import json
import re

from config import Config
from observability import Tracer
from core.llm import LLMClient
from .state import AgentState, SubTask

COMPLEXITY_PROMPT = """判断以下问题的复杂度，仅返回 JSON：
{{ "level": "simple" }} 或 {{ "level": "complex" }}

判断标准：
- simple：单一事实问答、简单计算、明确单步可回答的问题
- complex：需要多步推理、对比分析、综合多个信息源、涉及多个子话题

问题：{question}
"""


def make_complexity_check(llm: LLMClient, cfg: Config, tracer: Tracer):
    def node(state: AgentState) -> AgentState:
        question = state["question"]
        with tracer.span("complexity_check", question=question[:60]):
            res = llm.chat(
                [{"role": "user",
                  "content": COMPLEXITY_PROMPT.format(question=question)}],
                scope="complexity",
            )
        try:
            m = re.search(r"\{.*\}", res.content, re.S)
            data = json.loads(m.group(0) if m else res.content)
            level = data.get("level", "complex")
        except Exception:
            level = "complex"  # 解析失败保守走复杂路径

        trace = state.get("trace", [])
        trace.append(f"[complexity_check] 判定为 {level}")
        return {"complexity": level, "trace": trace}
    return node


def make_simple_subtask():
    """简单问题：直接创建单个 SubTask，跳过 Planner。"""
    def node(state: AgentState) -> AgentState:
        question = state["question"]
        subtasks = [SubTask(
            goal=question, status="pending", messages=[],
            reflections=[], result="", react_steps=0, reflection_count=0,
        )]
        trace = state.get("trace", [])
        trace.append("[simple_path] 跳过 Planner，直接作为单任务处理")
        return {"subtasks": subtasks, "current_index": 0, "trace": trace}
    return node
