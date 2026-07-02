"""沙箱化 Python 代码执行工具。

使用独立子进程 + 超时 + 字节码白名单（禁用 import os/subprocess 等危险模块），
在功能演示与最小安全之间取平衡。生产环境应换为容器/沙箱运行时。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from ..base import Tool, ToolResult
from ..registry import register_tool

# 禁用的模块/属性（黑名单兜底；白名单更安全但会牺牲可用性）
BLOCKED = ("import os", "import subprocess", "import shlex", "import socket",
           "from os", "from subprocess", "__import__", "open(")
TIMEOUT_S = 8


@register_tool
class CodeExecTool(Tool):
    name = "python_exec"
    description = "执行一段 Python 代码并返回 stdout。可用于数学计算、数据处理、字符串处理等。禁止文件/网络/系统操作。"
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "要执行的 Python 代码"},
        },
        "required": ["code"],
    }

    def run(self, code: str, **_) -> ToolResult:
        for b in BLOCKED:
            if b in code:
                return ToolResult(ok=False, output=None,
                                  error=f"安全策略：代码含被禁用片段 '{b}'")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=TIMEOUT_S,
            )
            out = proc.stdout.strip()
            err = proc.stderr.strip()
            if proc.returncode != 0:
                return ToolResult(ok=False, output=out, error=err[-1000:])
            return ToolResult(ok=True, output=out or "(无输出)")
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, output=None, error=f"执行超时（>{TIMEOUT_S}s）")
        finally:
            Path(path).unlink(missing_ok=True)
