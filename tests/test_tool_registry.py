"""工具注册机制测试（免 Key）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tools import list_tools, as_functions, execute, register_function
from core.tools.base import ToolResult


def test_builtin_tools_registered():
    names = {t.name for t in list_tools()}
    assert "python_exec" in names
    assert "web_search" in names
    # knowledge_search 在 RetrievalTool 注册时存在
    assert "knowledge_search" in names


def test_as_functions_format():
    fns = as_functions()
    assert any(f["function"]["name"] == "python_exec" for f in fns)
    fn = next(f for f in fns if f["function"]["name"] == "python_exec")
    assert "code" in fn["function"]["parameters"]["properties"]


def test_python_exec_runs():
    res = execute("python_exec", {"code": "print(1+1)"})
    assert isinstance(res, ToolResult)
    assert res.ok
    assert "2" in res.output


def test_python_exec_blocked():
    res = execute("python_exec", {"code": "import os; print(os.getcwd())"})
    assert not res.ok
    assert "安全策略" in res.error


def test_register_function():
    @register_function("double", "翻倍", {"type": "object",
                       "properties": {"x": {"type": "number"}}, "required": ["x"]})
    def double(x):
        return x * 2
    res = execute("double", {"x": 21})
    assert res.ok and res.output == 42


def test_unknown_tool():
    res = execute("nope", {})
    assert not res.ok
