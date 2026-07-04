"""消息序列合法性修复测试：防止 1214 错误回归。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.graph.react import _sanitize_messages, _trim_messages


def test_orphan_tool_removed():
    """截断后产生的孤儿 tool 消息应被丢弃。"""
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        # 这两个 tool 消息对应的 assistant 被截断了
        {"role": "tool", "tool_call_id": "call_A", "content": "r1"},
        {"role": "tool", "tool_call_id": "call_B", "content": "r2"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_C", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "call_D", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_C", "content": "r3"},
        {"role": "tool", "tool_call_id": "call_D", "content": "r4"},
    ]
    out = _sanitize_messages(msgs)
    # call_A / call_B 应被丢弃，call_C / call_D 保留
    tool_ids = [m["tool_call_id"] for m in out if m.get("role") == "tool"]
    assert "call_A" not in tool_ids
    assert "call_B" not in tool_ids
    assert "call_C" in tool_ids
    assert "call_D" in tool_ids


def test_valid_sequence_preserved():
    """合法序列应原样保留。"""
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_X", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_X", "content": "r"},
    ]
    out = _sanitize_messages(msgs)
    assert len(out) == len(msgs)


def test_trim_keeps_sequence_legal():
    """_trim_messages 截断后序列必须合法。"""
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "assistant", "content": "summary"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
    ]
    out = _trim_messages(msgs, max_tool_results=1)
    # 校验每个 tool 都有对应 assistant
    declared = set()
    for m in out:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                declared.add(tc.get("id"))
        if m.get("role") == "tool":
            assert m["tool_call_id"] in declared, f"孤儿 tool {m['tool_call_id']}"


def test_real_crash_dump_fixed():
    """复现真实崩溃场景：连续孤儿 tool 后跟合法块。"""
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "tool", "tool_call_id": "call_1", "content": "orphan1"},
        {"role": "tool", "tool_call_id": "call_2", "content": "orphan2"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "r"},
        {"role": "tool", "tool_call_id": "b", "content": "r"},
        {"role": "tool", "tool_call_id": "c", "content": "r"},
    ]
    out = _sanitize_messages(msgs)
    tool_ids = [m["tool_call_id"] for m in out if m.get("role") == "tool"]
    assert tool_ids == ["a", "b", "c"]


def test_ensure_at_least_one_user_message():
    """缺少 user 消息时（如 retry 截断后）应自动补一条。"""
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "x", "content": "r"},
    ]
    out = _sanitize_messages(msgs)
    assert any(m.get("role") == "user" for m in out)
    # user 应紧跟在 system 之后
    roles = [m.get("role") for m in out]
    assert roles.index("user") == roles.index("system") + 1
