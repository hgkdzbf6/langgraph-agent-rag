"""ReAct 推理节点：LLM + Function Calling 决策工具调用。

react_reason: 调 LLM（带工具声明），产出 tool_calls 或最终答复；
react_observe: 把工具结果整合回消息；若 LLM 已给出最终答复则标记子任务完成。
"""
from __future__ import annotations

import json

from config import Config
from observability import Tracer
from core.llm import LLMClient
from core.tools import as_functions
from .state import AgentState

SYSTEM = """你是一个能使用工具的 ReAct 智能体。针对当前子任务：
- 需要外部信息/计算时调用工具；
- 每次只决定下一步动作；
- 工具返回结果后，若已足以回答则给出最终答复（不要再调用工具）；
- 若工具失败，可换思路或重试。

子任务目标：{goal}
原始问题（参考上下文）：{question}
"""


def _init_messages(state: AgentState, sub) -> list[dict]:
    if not sub["messages"]:
        sub["messages"] = [
            {"role": "system",
             "content": SYSTEM.format(goal=sub["goal"], question=state["question"])},
            {"role": "user", "content": f"请完成子任务：{sub['goal']}"},
        ]
    return sub["messages"]


def _trim_messages(msgs: list[dict], max_tool_results: int = 3) -> list[dict]:
    """截断过长的 tool 结果，保留最近 N 轮的 tool 输出，避免 token 爆炸。

    策略：保留 system + 最近 2 条 user/assistant + 最近 max_tool_results 个 tool 结果，
    丢弃早期的 tool 结果（它们的内容已被 assistant 总结过）。
    """
    if len(msgs) <= 4:
        return msgs

    system = [m for m in msgs if m.get("role") == "system"]
    non_system = [m for m in msgs if m.get("role") != "system"]

    # 保留最后 N 条消息（含 assistant 对早期 tool 结果的总结）
    keep = max_tool_results * 2 + 2  # tool + assistant 配对
    if len(non_system) > keep:
        non_system = non_system[-keep:]

    return system + non_system


def make_react_reason(llm: LLMClient, cfg: Config, tracer: Tracer):
    def node(state: AgentState) -> AgentState:
        idx = state["current_index"]
        subs = state["subtasks"]
        sub = subs[idx]
        msgs = _init_messages(state, sub)
        # 截断过长的 tool 结果，防止 token 爆炸
        msgs = _trim_messages(msgs, max_tool_results=2)
        sub["messages"] = msgs
        with tracer.span("react_reason", subtask=idx, step=sub["react_steps"]):
            res = llm.chat_with_tools(msgs, as_functions(), scope=f"react#{idx}")
        # 追加 assistant 消息
        assistant: dict = {"role": "assistant", "content": res.content}
        if res.tool_calls:
            assistant["tool_calls"] = [tc.to_openai() for tc in res.tool_calls]
        msgs.append(assistant)
        sub["react_steps"] += 1
        sub["messages"] = msgs
        # 在 subtask 上挂一个临时槽，供 executor / observe 读取
        sub["_pending_tool_calls"] = [tc.__dict__ for tc in res.tool_calls]
        sub["_last_content"] = res.content
        return {"subtasks": subs}
    return node


def make_react_observe(cfg: Config, tracer: Tracer):
    """观察节点：判断是否继续工具循环、是否子任务完成。

    完成判定：最近一条 assistant 无 tool_calls 视为给出最终答复；
    或达到 max_react_steps 强制收尾（避免死循环）。
    """
    def node(state: AgentState) -> AgentState:
        idx = state["current_index"]
        subs = state["subtasks"]
        sub = subs[idx]
        last_asst = None
        for m in reversed(sub["messages"]):
            if m.get("role") == "assistant":
                last_asst = m
                break
        finished = last_asst is not None and not last_asst.get("tool_calls")
        if finished:
            sub["result"] = (last_asst or {}).get("content", "")
            sub["status"] = "done"
        elif sub["react_steps"] >= cfg.max_react_steps:
            # 步数上限，强制收尾
            sub["status"] = "done"
            sub["result"] = (last_asst or {}).get("content", "") or "(达到步数上限)"
        sub.pop("_pending_tool_calls", None)
        sub.pop("_last_content", None)
        return {"subtasks": subs}
    return node
