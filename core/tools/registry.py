"""MCP 风格工具注册中心。

- @register_tool 装饰器自动注册 Tool 子类；
- list_tools() 供 Agent 发现可用工具；
- as_functions() 转 function_call 格式交给 LLM；
- execute(name, args) 统一调用入口（带 Tracer span）。
"""
from __future__ import annotations

from typing import Any

from observability import Tracer
from .base import Tool, ToolResult, FunctionTool

_REGISTRY: dict[str, Tool] = {}


def register_tool(cls):
    """类装饰器：实例化并注册。"""
    if not issubclass(cls, Tool):
        raise TypeError(f"{cls} 必须继承 Tool")
    inst = cls()
    if not inst.name:
        raise ValueError(f"{cls.__name__} 未设置 name")
    _REGISTRY[inst.name] = inst
    return cls


def register_function(name: str, description: str, schema: dict):
    """函数装饰器：把普通函数注册为工具。"""
    def deco(fn):
        _REGISTRY[name] = FunctionTool(name, description, schema, fn)
        return fn
    return deco


def list_tools() -> list[Tool]:
    return list(_REGISTRY.values())


def get_tool(name: str) -> Tool | None:
    return _REGISTRY.get(name)


def as_functions() -> list[dict]:
    return [t.as_function() for t in _REGISTRY.values()]


def execute(name: str, arguments: dict[str, Any], tracer: Tracer | None = None,
            scope: str | None = None) -> ToolResult:
    """统一调用入口；tracer 存在时记录 span。"""
    tool = _REGISTRY.get(name)
    if tool is None:
        return ToolResult(ok=False, output=None, error=f"未知工具: {name}")
    sc = scope or f"tool:{name}"
    if tracer is None:
        return tool.run(**arguments)
    with tracer.span(sc, tool=name, args_keys=list(arguments.keys())):
        try:
            res = tool.run(**arguments)
            return res
        except Exception as e:
            return ToolResult(ok=False, output=None, error=f"{type(e).__name__}: {e}")
