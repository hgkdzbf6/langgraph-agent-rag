"""端到端演示：ReAct + Reflection + 长程规划 + RAG，全程可观测。

用法：
  1. 复制 .env.example 为 .env，填入 ZHIPU_API_KEY / ZCODE_BASE_URL
  2. python main.py "你的复杂问题"
  3. 不带参数则用内置示例问题

环境未配置 API Key 时会报错并提示；可用 python tests/ 下的 mock 测试离线验证框架。
"""
from __future__ import annotations

import sys

from config import CONFIG
from observability import Tracer, CostTracker
from core.llm import GLMClient
from core.tools import builtin  # noqa: F401  触发工具注册
from core.tools.builtin.retrieval import set_pipeline
from core.graph import build_graph
from rag import build_pipeline


def main(question: str | None = None) -> None:
    if not CONFIG.llm.configured:
        print("[ERROR] 未检测到 API Key。请在 .env 设置 ZHIPU_API_KEY / ZCODE_BASE_URL。")
        print("        参考 .env.example。离线验证请运行: python -m pytest tests/")
        sys.exit(1)

    tracer = Tracer()
    cost = CostTracker(model=CONFIG.llm.model)

    # 1. 构建并装载 RAG 知识库
    print("==== 构建 RAG 知识库 ====")
    rag = build_pipeline(CONFIG, tracer)
    n = rag.ingest_dir("data/knowledge")
    print(f"已索引 {n} 个 chunk")
    set_pipeline(rag)

    # 2. 构建 LLM 客户端与 Agent 图
    llm = GLMClient(CONFIG.llm, tracer, cost)
    graph = build_graph(llm, CONFIG, tracer)

    q = question or "请结合知识库说明：这个自研 Agent 框架具备哪些核心能力，RAG 如何优化召回质量，以及它相对 LangChain 的取舍是什么？"
    print(f"\n==== 问题 ====\n{q}\n")

    # 3. 运行
    state = {"question": q}
    final = graph.invoke(state, config={"recursion_limit": 60})

    # 4. 输出结果 + 可观测性报告
    print("\n==== 最终答案 ====")
    print(final.get("final_answer", "(无)"))

    print("\n==== 任务轨迹 ====")
    for t in final.get("trace", []):
        print(" ", t)

    print("\n==== 子任务 ====")
    for i, s in enumerate(final.get("subtasks", [])):
        print(f"  [{i}] {s['goal']}  -> {s['status']}  (react {s['react_steps']}步, "
              f"反思 {s['reflection_count']}次)")

    print("\n==== 链路追踪 ====")
    print(tracer.tree())

    print("\n" + cost.report())


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else None
    main(q)
