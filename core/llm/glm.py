"""智谱 GLM Provider —— 通过 OpenAI 兼容接口调用 coding-plan 网关。

复用 openai SDK（base_url 指向网关），所有调用都打 Tracer span 并写入 CostTracker。
"""
from __future__ import annotations

from typing import Any

from openai import OpenAI

from config import LLMConfig
from observability import Tracer, CostTracker
from .base import LLMClient, ChatResult, ToolCall
from .cache import LLMCache


class GLMClient(LLMClient):
    def __init__(self, cfg: LLMConfig, tracer: Tracer, cost: CostTracker,
                 cache: LLMCache | None = None) -> None:
        if not cfg.configured:
            raise RuntimeError(
                "GLMClient 未配置：请设置 ZCODE_BASE_URL / ZHIPU_API_KEY（见 .env.example）"
            )
        self.cfg = cfg
        self.tracer = tracer
        self.cost = cost
        self.cache = cache
        # thinking 可被 set_thinking 动态切换（由 complexity 节点按难度设置）
        self._thinking: str = cfg.thinking
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def set_thinking(self, mode: str) -> None:
        """动态切换思维链模式：'disabled' 关闭（快）/ 'enabled' 开启（准）。

        由 complexity_check 节点按问题难度调用：
        simple/medium → disabled，hard → enabled。
        """
        if mode in ("disabled", "enabled"):
            self._thinking = mode

    def _call(self, messages: list[dict], tools: list[dict] | None,
              temperature: float | None, max_tokens: int | None, scope: str) -> ChatResult:
        # 检查缓存
        if self.cache is not None:
            cached = self.cache.get(self.cfg.model, messages, tools)
            if cached is not None:
                with self.tracer.span(f"llm:{scope}", model=self.cfg.model,
                                      cached=True, n_msgs=len(messages)):
                    pass
                return cached

        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
        }
        # thinking 模式：按当前动态设置（默认 disabled；hard 问题由 complexity 节点切 enabled）
        if self._thinking in ("disabled", "enabled"):
            kwargs["extra_body"] = {"thinking": {"type": self._thinking}}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        with self.tracer.span(f"llm:{scope}", model=self.cfg.model, n_msgs=len(messages)):
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                # 报错时 dump messages 到文件，便于排查 1214 等参数错误
                import json as _json, time as _time
                dump = {"scope": scope, "error": str(e)[:500],
                        "kwargs_keys": list(kwargs.keys()),
                        "messages": messages,
                        "tools": tools}
                try:
                    from pathlib import Path as _P
                    _P("data/_llm_error_dump.json").write_text(
                        _json.dumps(dump, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
                except Exception:
                    pass
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

        result = ChatResult(content=choice.content or "", tool_calls=tool_calls,
                            usage=usage, raw=resp)

        # 写入缓存（仅缓存无 tool_calls 的纯文本响应，避免缓存工具调用导致状态不一致）
        if self.cache is not None and not tool_calls:
            self.cache.put(self.cfg.model, messages, result, tools)

        return result

    def chat(self, messages, *, temperature=None, max_tokens=None, scope="llm") -> ChatResult:
        return self._call(messages, None, temperature, max_tokens, scope)

    def chat_with_tools(self, messages, tools, *, temperature=None,
                        max_tokens=None, scope="llm") -> ChatResult:
        return self._call(messages, tools, temperature, max_tokens, scope)
