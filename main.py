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
from core.llm.cache import LLMCache
from core.tools import builtin  # noqa: F401  触发工具注册
from core.tools.builtin.retrieval import set_pipeline, clear_cache
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
    knowledge_dir = CONFIG.rag.knowledge_dir
    print(f"==== 构建 RAG 知识库 ({knowledge_dir}) ====")
    rag = build_pipeline(CONFIG, tracer)
    n = rag.ingest_dir(knowledge_dir)
    print(f"已索引 {n} 个 chunk")
    set_pipeline(rag)
    clear_cache()

    # 2. 构建 LLM 客户端与 Agent 图
    cache = LLMCache() if CONFIG.enable_llm_cache else None
    llm = GLMClient(CONFIG.llm, tracer, cost, cache=cache)
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


def run_eval():
    """运行评测套件。"""
    from eval import RAGEvaluator, ReflectionEvaluator, E2EEvaluator

    if not CONFIG.llm.configured:
        print("[ERROR] 评测需要 API Key。请在 .env 设置 ZHIPU_API_KEY / ZCODE_BASE_URL。")
        sys.exit(1)

    tracer = Tracer()
    cost = CostTracker(model=CONFIG.llm.model)
    rag = build_pipeline(CONFIG, tracer)
    n = rag.ingest_dir("data/knowledge")
    print(f"已索引 {n} 个 chunk")

    print("\n" + "=" * 50)
    print("1. RAG 召回质量评测")
    print("=" * 50)
    rag_eval = RAGEvaluator(rag)
    print(rag_eval.report())

    print("\n" + "=" * 50)
    print("2. Reflection 收益评测")
    print("=" * 50)
    llm = GLMClient(CONFIG.llm, tracer, cost)
    ref_eval = ReflectionEvaluator(llm)
    print(ref_eval.report())

    print("\n" + "=" * 50)
    print("3. 端到端评测")
    print("=" * 50)
    def llm_factory(cfg, tr, ct):
        from core.llm.glm import GLMClient as GLMC
        return GLMC(cfg.llm, tr, ct)
    e2e_eval = E2EEvaluator(llm_factory)
    print(e2e_eval.report())


if __name__ == "__main__":
    if "--eval" in sys.argv:
        run_eval()
    else:
        q = sys.argv[1] if len(sys.argv) > 1 else None
        main(q)
