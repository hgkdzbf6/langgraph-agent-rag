"""轻量链路追踪：每个节点/工具/LLM 调用记一条 span，组成树。

不依赖 OpenTelemetry，零额外依赖；run 结束后可打印 trace 树或导出 JSON，
便于排查 ReAct 循环、Reflection 回退、工具并发等行为。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Span:
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: Optional[str] = None
    start: float = field(default_factory=time.time)
    end: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"          # ok / error
    error: Optional[str] = None

    @property
    def elapsed_ms(self) -> float:
        return round(((self.end or time.time()) - self.start) * 1000, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["elapsed_ms"] = self.elapsed_ms
        return d


class Tracer:
    """线程安全（用 GIL 简化）的 span 收集器，支持父子层级。"""

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._stack: list[str] = []

    def start(self, name: str, **attributes: Any) -> Span:
        parent = self._stack[-1] if self._stack else None
        span = Span(name=name, parent_id=parent, attributes=attributes)
        self.spans.append(span)
        self._stack.append(span.span_id)
        return span

    def end(self, span: Span, status: str = "ok", error: Optional[str] = None,
            **extra_attrs: Any) -> None:
        span.end = time.time()
        span.status = status
        span.error = error
        span.attributes.update(extra_attrs)
        if self._stack and self._stack[-1] == span.span_id:
            self._stack.pop()

    # 上下文管理器语法糖：with tracer.span("node") as s: ...
    class _Ctx:
        def __init__(self, tracer: "Tracer", name: str, attrs: dict):
            self.tracer = tracer
            self.name = name
            self.attrs = attrs
            self.span: Optional[Span] = None

        def __enter__(self) -> Span:
            self.span = self.tracer.start(self.name, **self.attrs)
            return self.span

        def __exit__(self, exc_type, exc, _tb):
            if exc is not None:
                self.tracer.end(self.span, status="error",
                                error=f"{exc_type.__name__}: {exc}")
            else:
                self.tracer.end(self.span)
            return False

    def span(self, name: str, **attrs: Any):
        return Tracer._Ctx(self, name, attrs)

    def tree(self) -> str:
        """把 spans 渲染成缩进树。"""
        by_parent: dict[Optional[str], list[Span]] = {}
        for s in self.spans:
            by_parent.setdefault(s.parent_id, []).append(s)
        lines: list[str] = []

        def render(parent: Optional[str], depth: int) -> None:
            for s in by_parent.get(parent, []):
                tag = "✗" if s.status == "error" else "✓"
                lines.append(f"{'  ' * depth}{tag} {s.name}  [{s.elapsed_ms} ms]"
                             + (f"  err={s.error}" if s.error else ""))
                render(s.span_id, depth + 1)
        render(None, 0)
        return "\n".join(lines)

    def to_json(self) -> list[dict]:
        return [s.to_dict() for s in self.spans]
