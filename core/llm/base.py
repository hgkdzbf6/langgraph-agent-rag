"""LLM 客户端抽象。

统一 chat / chat_with_tools 两个方法，屏蔽不同 Provider 差异；
ChatResult 携带 content、tool_calls、原始 usage，供 CostTracker 归因。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai(self) -> dict:
        import json
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)},
        }


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)  # prompt/completion/total_tokens
    raw: Any = None


class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], *, temperature: float | None = None,
             max_tokens: int | None = None, scope: str = "llm") -> ChatResult:
        ...

    @abstractmethod
    def chat_with_tools(self, messages: list[dict], tools: list[dict], *,
                        temperature: float | None = None,
                        max_tokens: int | None = None,
                        scope: str = "llm") -> ChatResult:
        ...
