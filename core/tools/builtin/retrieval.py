"""RAG 检索工具：从知识库检索相关片段。

依赖 rag.pipeline.RAGPipeline —— 通过 set_pipeline 注入，避免循环导入。
"""
from __future__ import annotations

from typing import Any

from ..base import Tool, ToolResult
from ..registry import register_tool

_PIPELINE = None


def set_pipeline(pipeline) -> None:
    """在 main.py 启动时注入已构建的 RAG pipeline。"""
    global _PIPELINE
    _PIPELINE = pipeline


@register_tool
class RetrievalTool(Tool):
    name = "knowledge_search"
    description = "在本地知识库中检索与问题相关的文档片段。输入自然语言查询，返回最相关的若干片段。"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索查询语句"},
            "topn": {"type": "integer", "description": "返回片段数，默认5", "default": 5},
        },
        "required": ["query"],
    }

    def run(self, query: str, topn: int = 5, **_: Any) -> ToolResult:
        if _PIPELINE is None:
            return ToolResult(ok=False, output=None, error="知识库未初始化")
        chunks = _PIPELINE.query(query, topn=topn)
        if not chunks:
            return ToolResult(ok=True, output="（未检索到相关内容）")
        return ToolResult(ok=True, output=chunks)
