"""Web 搜索工具。

优先调真实搜索 API；若未配置 API 则降级为 stub，返回提示信息，
保证框架在离线/无 Key 环境仍可演示 ReAct 流程。
"""
from __future__ import annotations

import os

from ..base import Tool, ToolResult
from ..registry import register_tool

SEARCH_API = os.getenv("WEB_SEARCH_API", "")   # 如有搜索 API key 填这里
SEARCH_URL = os.getenv("WEB_SEARCH_URL", "")


@register_tool
class WebSearchTool(Tool):
    name = "web_search"
    description = "在互联网上搜索实时信息（新闻、文档、数据等）。返回摘要式结果列表。"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
        },
        "required": ["query"],
    }

    def run(self, query: str, **_) -> ToolResult:
        if SEARCH_API and SEARCH_URL:
            try:
                import requests
                resp = requests.get(
                    SEARCH_URL, params={"q": query, "api_key": SEARCH_API}, timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                return ToolResult(ok=True, output=str(data)[:2000])
            except Exception as e:
                return ToolResult(ok=False, output=None,
                                  error=f"搜索接口异常: {type(e).__name__}: {e}")
        # 离线兜底：返回结构化占位结果，使流程可继续
        return ToolResult(
            ok=True,
            output=(f"[stub web_search] 未配置搜索 API，模拟结果。query='{query}'。"
                    "如需真实检索，请在 .env 设置 WEB_SEARCH_API / WEB_SEARCH_URL。"),
        )
