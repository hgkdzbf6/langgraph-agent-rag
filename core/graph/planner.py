"""长程任务规划节点：把复杂问题拆解为有序子任务。

用 LLM + 结构化输出指令生成 JSON 子任务列表，失败时降级为单任务。
"""
from __future__ import annotations

import json
import re

from config import Config
from observability import Tracer
from core.llm import LLMClient
from .state import AgentState, SubTask

PLAN_PROMPT = """你是一个任务规划器。请把用户问题拆解为 **尽量少** 的子任务（通常 1-3 个足够）。

原则：
- 能 1 个子任务完成的不要拆 2 个
- 只在确实需要不同信息源/不同分析维度时才拆分
- 每个子任务应是独立的、可一次检索+推理完成的小目标

仅输出 JSON 数组，不要任何解释文字，格式：
[
  {{ "goal": "子任务描述" }}
]

最多 {n} 个子任务。用户问题：{question}
"""


def make_planner(llm: LLMClient, cfg: Config, tracer: Tracer):
    def node(state: AgentState) -> AgentState:
        question = state["question"]
        with tracer.span("planner", question=question[:60]):
            res = llm.chat(
                [{"role": "user",
                  "content": PLAN_PROMPT.format(n=cfg.max_subtasks, question=question)}],
                scope="planner",
            )
        subtasks: list[SubTask] = []
        try:
            # 兼容模型偶尔带 ```json fence
            text = re.search(r"\[.*\]", res.content, re.S)
            raw = json.loads(text.group(0) if text else res.content)
            for item in raw[: cfg.max_subtasks]:
                subtasks.append(SubTask(
                    goal=item.get("goal", str(item)),
                    status="pending", messages=[], reflections=[],
                    result="", react_steps=0, reflection_count=0,
                ))
        except Exception:
            # 降级：单任务
            subtasks.append(SubTask(
                goal=question, status="pending", messages=[],
                reflections=[], result="", react_steps=0, reflection_count=0,
            ))
        if not subtasks:
            subtasks.append(SubTask(
                goal=question, status="pending", messages=[],
                reflections=[], result="", react_steps=0, reflection_count=0,
            ))
        trace = state.get("trace", [])
        trace.append(f"[planner] 拆解为 {len(subtasks)} 个子任务")
        return {"subtasks": subtasks, "current_index": 0, "trace": trace}
    return node
