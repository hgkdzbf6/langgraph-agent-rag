"""复杂度前置判断节点：区分简单/中等/复杂问题，避免简单问题走完整 Planner 流程。

三档难度同时决定后续 LLM 调用是否开启 thinking 思维链：
- simple：单一事实/计算，无需深度推理 → thinking 关闭（快）
- medium：检索+总结、单领域问答 → thinking 关闭（快，检索为主）
- hard：多步推理、对比分析、跨领域综合 → thinking 开启（准）
"""
from __future__ import annotations

import json
import re

from config import Config
from observability import Tracer
from core.llm import LLMClient
from .state import AgentState, SubTask

COMPLEXITY_PROMPT = """判断以下问题的复杂度，仅返回 JSON：
{{ "level": "simple" | "medium" | "hard" }}

判断标准：
- simple：单一事实问答、简单计算、明确单步可回答（如"X是什么""算个数字"）
- medium：需要检索后总结、单领域内的问答、描述性说明（如"总结X的工作""X有哪些功能"）
- hard：需要多步推理、对比分析优劣、跨领域综合、给出决策建议（如"对比X和Y并建议""分析根因并给出方案"）

问题：{question}
"""

# 难度 → 是否开启 thinking 的映射
_THINKING_BY_LEVEL = {
    "simple": "disabled",
    "medium": "disabled",
    "hard": "enabled",
}


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
            level = data.get("level", "medium")
            if level not in ("simple", "medium", "hard"):
                level = "medium"
        except Exception:
            level = "medium"  # 解析失败走中等路径（兼顾速度与质量）

        # 根据难度动态切换 thinking（影响后续所有 LLM 调用）
        if cfg.adaptive_thinking:
            thinking = _THINKING_BY_LEVEL.get(level, "disabled")
            if hasattr(llm, "set_thinking"):
                llm.set_thinking(thinking)

        trace = state.get("trace", [])
        thinking_state = _THINKING_BY_LEVEL.get(level, "disabled")
        trace.append(f"[complexity_check] 判定为 {level}（thinking={thinking_state}）")
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
