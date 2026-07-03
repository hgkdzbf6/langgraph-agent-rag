"""Agent 状态定义（LangGraph State）。

承载：原始问题、规划出的子任务、当前进度、消息历史、反思、轨迹等。
所有节点读写同一份 state，由 LangGraph 负责流转。
"""
from __future__ import annotations

from typing import Any, TypedDict


class SubTask(TypedDict):
    goal: str               # 子任务目标
    status: str             # pending / running / done / failed
    messages: list[dict]    # 该子任务的对话历史（含 tool 调用/结果）
    reflections: list[str]  # 反思记录
    result: str             # 子任务结论
    react_steps: int        # 已执行 ReAct 轮次
    reflection_count: int   # 已纠错次数


class AgentState(TypedDict, total=False):
    question: str                   # 原始长程任务
    subtasks: list[SubTask]         # 规划出的子任务
    current_index: int              # 正在处理的子任务下标
    final_answer: str               # 最终聚合答案
    trace: list[str]                # 高层轨迹（人类可读）
    done: bool
    complexity: str                 # "simple" | "complex"，由 complexity_check 节点设置
