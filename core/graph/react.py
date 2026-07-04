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
- 重要：一旦工具返回的信息已足以回答子任务，必须立即停止调用工具，直接给出最终答复文本；
- 不要重复调用相同或相似的工具；不要为了"再多查一点"而持续调用；
- 若工具失败，可换思路或基于已有信息作答。

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

    策略：保留 system + 最近的消息，并**保证消息序列对 LLM API 合法**：
    - 每个 role=tool 消息前面必须有声明了对应 tool_call_id 的 assistant(tool_calls)；
    - 截断点不能落在 assistant(tool_calls) 与其 tool 结果之间。
    """
    if len(msgs) <= 4:
        return msgs

    system = [m for m in msgs if m.get("role") == "system"]
    non_system = [m for m in msgs if m.get("role") != "system"]

    keep = max_tool_results * 2 + 2  # tool + assistant 配对
    if len(non_system) > keep:
        non_system = non_system[-keep:]

    trimmed = system + non_system
    return _sanitize_messages(trimmed)


def _sanitize_messages(msgs: list[dict]) -> list[dict]:
    """修复截断后可能出现的非法消息序列。

    规则（智谱/OpenAI 等 function calling API 的通用约束）：
    - role=tool 消息必须紧跟在声明了对应 tool_call_id 的 assistant 消息之后；
    - 若 tool 消息找不到对应的 assistant，则丢弃（避免孤儿 tool 引发 1214）；
    - 必须至少有一条 user 消息（智谱要求），否则在 system 后补一条引导 user 消息。
    """
    # 收集所有已声明的 tool_call_id（来自 assistant 消息且位于 tool 消息之前）
    out: list[dict] = []
    declared_ids: set[str] = set()
    for m in msgs:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tcid = tc.get("id")
                if tcid:
                    declared_ids.add(tcid)
            out.append(m)
        elif role == "tool":
            tcid = m.get("tool_call_id")
            if tcid in declared_ids and out and out[-1].get("role") in ("assistant", "tool"):
                if _has_recent_assistant_with_id(out, tcid):
                    out.append(m)
                # 否则丢弃孤儿 tool
            # 否则丢弃孤儿 tool
        else:
            declared_ids = set()
            out.append(m)

    # 保证至少有一条 user 消息（智谱 API 要求）
    has_user = any(m.get("role") == "user" for m in out)
    if not has_user:
        # 在 system 之后插入一条引导消息
        for i, m in enumerate(out):
            if m.get("role") == "system":
                out.insert(i + 1, {"role": "user", "content": "请继续完成任务。"})
                break
        else:
            out.insert(0, {"role": "user", "content": "请继续完成任务。"})
    return out


def _has_recent_assistant_with_id(msgs: list[dict], tcid: str) -> bool:
    """从末尾向前找，跳过连续的 tool 消息，看是否能遇到声明了 tcid 的 assistant。"""
    for m in reversed(msgs):
        role = m.get("role")
        if role == "tool":
            continue
        if role == "assistant" and m.get("tool_calls"):
            return any(tc.get("id") == tcid for tc in m["tool_calls"])
        # 遇到其他角色说明断链了
        return False
    return False


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


def make_react_observe(cfg: Config, tracer: Tracer, llm: LLMClient | None = None):
    """观察节点：判断是否继续工具循环、是否子任务完成。

    完成判定：最近一条 assistant 无 tool_calls 视为给出最终答复；
    或达到 max_react_steps 强制收尾（避免死循环）。
    强制收尾时若有 llm，会再做一次"基于已有信息总结"的调用，避免直接丢弃结果。
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
            # 步数上限，强制收尾：尝试让 LLM 基于已有信息总结
            summary = ""
            if llm is not None:
                with tracer.span("force_summarize", subtask=idx):
                    try:
                        force_msgs = list(sub["messages"]) + [
                            {"role": "user",
                             "content": "已达到工具调用次数上限。请不要再调用任何工具，"
                                        "直接根据目前已掌握的信息，给出对该子任务的最终答复。"}
                        ]
                        r = llm.chat(force_msgs, scope=f"force#{idx}")
                        summary = r.content
                    except Exception:
                        summary = ""
            sub["status"] = "done"
            sub["result"] = summary or (last_asst or {}).get("content", "") or "(达到步数上限，未能得出结论)"
        sub.pop("_pending_tool_calls", None)
        sub.pop("_last_content", None)
        return {"subtasks": subs}
    return node
