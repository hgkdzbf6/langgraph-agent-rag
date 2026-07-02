# 自研 Agent 调度框架 + RAG 知识库

基于 **Python + LangGraph** 的轻量 Agent 框架：ReAct 推理 + Reflection 自我纠错 +
长程任务规划 + 多工具并发编排，MCP 风格工具注册，配套 RAG（FAISS + 重排）与全链路
可观测性（span 追踪 + token 成本统计）。LLM 走**智谱 GLM coding-plan 网关**（OpenAI 兼容）。

详细设计见 [DESIGN.md](./DESIGN.md)。

## 特性
- 🧠 **ReAct + Reflection + 规划**：长程任务拆子任务、逐个 ReAct、反思纠错、汇总
- 🔧 **MCP 风格工具**：`@register_tool` 注册即用；内置检索/搜索/代码执行
- ⚡ **并发工具调用**：同一轮多 `tool_calls` 并发执行
- 📚 **RAG**：递归切分 → embedding 缓存 → FAISS → 重排（API 或 MMR 兜底）
- 📊 **可观测**：span 链路树 + token 成本按 node/tool 归因
- 🪶 **轻依赖**：核心仅 langgraph/openai/faiss/numpy，无 torch

## 快速开始

```bash
# 1. 安装依赖
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. 配置 API（复制并填入智谱 Key）
cp .env.example .env
#   编辑 .env：ZHIPU_API_KEY=...  ZCODE_BASE_URL=...

# 3. 端到端演示（真实调用）
.venv/bin/python main.py
# 或指定问题
.venv/bin/python main.py "请说明本框架相对 LangChain 的取舍"

# 4. 离线测试（免 Key，mock LLM/embedder）
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -v
```

## 环境变量（.env）

| 变量 | 说明 |
|---|---|
| `ZCODE_BASE_URL` / `ZAI_BUSINESS_BASE_URL` | coding-plan 网关地址（OpenAI 兼容） |
| `ZHIPU_API_KEY` / `ZCODE_API_KEY` | 智谱 API Key |
| `GLM_MODEL` | 模型名，默认 `GLM-5.2`（可切 `GLM-4.5`） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | RAG 切分参数 |
| `RETRIEVE_TOPK` / `RERANK_TOPN` | 召回/重排数量 |

## 示例输出
运行后可看到：最终答案、任务轨迹、子任务统计、链路追踪树、token 成本报告。

## 无 API Key 怎么办？
离线测试完全可跑（`pytest tests/`，用 mock LLM/embedder 验证框架正确性）。
真实端到端调用需一个智谱 API Key；填入 `.env` 后即可。
