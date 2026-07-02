"""智谱 GLM Provider —— 通过 OpenAI 兼容接口调用 coding-plan 网关。

复用 openai SDK（base_url 指向网关），所有调用都打 Tracer span 并写入 CostTracker。
"""
from __future__ import annotations

from typing import Any

from openai import OpenAI

from config import LLMConfig
from observability import Tracer, CostTracker
from .base import LLMClient, ChatResult, ToolCall


class GLMClient(LLMClient):
    def __init__(self, cfg: LLMConfig, tracer: Tracer, cost: CostTracker) -> None:
        if not cfg.configured:
            raise RuntimeError(
                "GLMClient 未配置：请设置 ZCODE_BASE_URL / ZHIPU_API_KEY（见 .env.example）"
            )
        self.cfg = cfg
        self.tracer = tracer
        self.cost = cost
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def _call(self, messages: list[dict], tools: list[dict] | None,
              temperature: float | None, max_tokens: int | None, scope: str) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        with self.tracer.span(f"llm:{scope}", model=self.cfg.model, n_msgs=len(messages)):
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                # tracer 的 with 退出会自动打 error span
                raise

        choice = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        if getattr(choice, "tool_calls", None):
            import json
            for tc in choice.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                "total_tokens": getattr(resp.usage, "total_tokens", 0),
            }
            self.cost.add(usage, scope=scope, model=self.cfg.model)

        return ChatResult(content=choice.content or "", tool_calls=tool_calls,
                          usage=usage, raw=resp)

    def chat(self, messages, *, temperature=None, max_tokens=None, scope="llm") -> ChatResult:
        return self._call(messages, None, temperature, max_tokens, scope)

    def chat_with_tools(self, messages, tools, *, temperature=None,
                        max_tokens=None, scope="llm") -> ChatResult:
        return self._call(messages, tools, temperature, max_tokens, scope)
