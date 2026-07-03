"""沙箱化 Python 代码执行工具。

支持两种模式：
- local：子进程 + 黑名单过滤（当前默认，仅适合演示）
- docker：Docker 容器隔离（推荐生产使用，进程/网络/文件系统三重隔离）

黑名单作为第一层快速检查，Docker 模式提供真正的安全边界。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

from ..base import Tool, ToolResult
from ..registry import register_tool

# 禁用的模块/属性（黑名单兜底；白名单更安全但会牺牲可用性）
BLOCKED = ("import os", "import subprocess", "import shlex", "import socket",
           "from os", "from subprocess", "__import__", "open(")
TIMEOUT_S = 8


def _check_blacklist(code: str) -> str | None:
    """检查黑名单，返回被禁片段或 None。"""
    for b in BLOCKED:
        if b in code:
            return b
    return None


def _run_local(code: str) -> ToolResult:
    """本地子进程执行。"""
    blocked = _check_blacklist(code)
    if blocked:
        return ToolResult(ok=False, output=None,
                          error=f"安全策略：代码含被禁用片段 '{blocked}'")
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


def _run_docker(code: str, image: str, timeout: int,
                memory: str, cpus: float) -> ToolResult:
    """Docker 容器隔离执行。

    安全边界：
    - --network=none：禁止网络访问
    - --memory + --cpus：资源限制
    - --read-only：只读文件系统（/tmp 除外）
    - --security-opt no-new-privileges：禁止提权
    - --rm：容器退出后自动清理
    """
    # 黑名单仍作为快速检查
    blocked = _check_blacklist(code)
    if blocked:
        return ToolResult(ok=False, output=None,
                          error=f"安全策略：代码含被禁用片段 '{blocked}'")

    # 检查 docker 是否可用
    if not shutil.which("docker"):
        return ToolResult(ok=False, output=None,
                          error="Docker 不可用：请安装 Docker 或切换到 sandbox_mode='local'")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        host_path = f.name

    try:
        cmd = [
            "docker", "run", "--rm",
            "--network=none",
            f"--memory={memory}",
            f"--cpus={cpus}",
            "--read-only",
            "--tmpfs", "/tmp:size=64m",
            "--security-opt", "no-new-privileges",
            "--stop-timeout", str(timeout),
            image,
            "python", "/code/script.py",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout + 5,  # 给 Docker 启动留余量
            volumes={host_path: {"bind": "/code/script.py", "mode": "ro"}},
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        if proc.returncode != 0:
            return ToolResult(ok=False, output=out, error=err[-1000:])
        return ToolResult(ok=True, output=out or "(无输出)")
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, output=None,
                          error=f"Docker 执行超时（>{timeout}s）")
    except Exception as e:
        return ToolResult(ok=False, output=None,
                          error=f"Docker 异常：{type(e).__name__}: {e}")
    finally:
        Path(host_path).unlink(missing_ok=True)


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

    def __init__(self):
        from config import CONFIG
        self._sandbox_mode = CONFIG.sandbox_mode
        self._docker_image = CONFIG.docker_image
        self._docker_timeout = CONFIG.docker_timeout
        self._docker_memory = CONFIG.docker_memory
        self._docker_cpus = CONFIG.docker_cpus

    def run(self, code: str, **_) -> ToolResult:
        if self._sandbox_mode == "docker":
            return _run_docker(code, self._docker_image, self._docker_timeout,
                               self._docker_memory, self._docker_cpus)
        return _run_local(code)
