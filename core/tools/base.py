"""MCP 风格的 Tool 基类。

与 MCP tool 定义对齐：name / description / input_schema(JSON Schema) / handler。
新增工具只需继承 Tool 并用 @register_tool 注册。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolResult:
    ok: bool
    output: Any            # 序列化为 str 后喂回 LLM
    error: str | None = None

    def to_str(self) -> str:
        if self.ok:
            return str(self.output)
        return f"[ERROR] {self.error}"


class Tool(ABC):
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {}   # JSON Schema

    @abstractmethod
    def run(self, **arguments: Any) -> ToolResult:
        ...

    def as_function(self) -> dict:
        """转 OpenAI/智谱 function_call 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


# 支持把普通函数快速包装成 Tool
class FunctionTool(Tool):
    def __init__(self, name: str, description: str, schema: dict, fn: Callable[..., Any]):
        self.name = name
        self.description = description
        self.input_schema = schema
        self._fn = fn

    def run(self, **arguments: Any) -> ToolResult:
        try:
            return ToolResult(ok=True, output=self._fn(**arguments))
        except Exception as e:
            return ToolResult(ok=False, output=None, error=f"{type(e).__name__}: {e}")
