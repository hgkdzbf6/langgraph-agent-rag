# 自研 Agent 调度框架 —— 架构设计文档

> 基于 Python + LangGraph 的轻量 Agent 框架，集成 ReAct 推理、Reflection 自我纠错、
> 长程任务规划、多工具并发编排、MCP 风格工具注册，并配套 RAG 知识库与全链路可观测性。

---

## 1. 设计目标

| 目标 | 说明 |
|---|---|
| **可控性** | 用 LangGraph 显式状态图编排，节点/边/条件一目了然，便于调试与改造 |
| **可观测性** | 内建链路追踪（span 树）+ token 成本归因（按 node/tool/run 维度） |
| **轻依赖** | 核心仅 `langgraph / openai / faiss / numpy`，不引入 torch |
| **可扩展** | 新工具零侵入注册；新 LLM Provider 单类替换 |
| **成本友好** | Embedding hash 缓存；rerank 回退 MMR；并发降低延迟 |

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py (入口)                       │
├─────────────────────────────────────────────────────────────┤
│   LangGraph StateGraph                                       │
│   START → planner → [子任务循环] → aggregator → END          │
│                      │                                       │
│                      ▼                                       │
│        ┌─────────────────────────────────┐                  │
│        │ react_reason (LLM+FunctionCall) │◄──┐ retry         │
│        └────────────┬────────────────────┘   │               │
│                     ▼                          │               │
│        ┌─────────────────────────────────┐   │               │
│        │ tool_executor (并发执行)         │    │               │
│        └────────────┬────────────────────┘   │               │
│                     ▼                          │               │
│        ┌─────────────────────────────────┐   │               │
│        │ react_observe (完成判定)         │    │               │
│        └────────────┬────────────────────┘   │               │
│                     ▼ done                    │               │
│        ┌─────────────────────────────────┐   │               │
│        │ reflector (自我纠错)             │────┘               │
│        └────────────┬────────────────────┘   ok               │
│                     ▼                          │               │
│                 advance (下一子任务)           │               │
└─────────────────────────────────────────────────────────────┘
        │                                  │
        ▼                                  ▼
┌──────────────────┐               ┌──────────────────┐
│ core/tools       │               │ rag/             │
│ MCP 风格注册中心  │               │ chunk→embed→     │
│ retrieval/search │◄──────────────│ FAISS→rerank     │
│ /code_exec       │   检索工具调用 │                  │
└──────────────────┘               └──────────────────┘
        │                                  │
        ▼                                  ▼
┌──────────────────────────────────────────────────────────────┐
│ observability: Tracer(span 树) + CostTracker(token 归因)      │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ core/llm: LLMClient 抽象 → GLMClient (智谱 coding-plan 网关) │
└──────────────────────────────────────────────────────────────┘
```

## 3. 核心机制

### 3.1 LangGraph 状态图
- `AgentState`（TypedDict）承载：question、subtasks、current_index、final_answer、trace、done；
- 每个子任务 `SubTask` 维护自己的 messages、reflections、react_steps、reflection_count；
- **所有状态变更通过节点返回 dict 体现**，条件路由函数只读不写（关键正确性约束）；
- 条件边实现 ReAct 循环（observe → reason）与 Reflection 回退（reflector → reason）。

### 3.2 ReAct 推理（core/graph/react.py）
- `react_reason`：把 system prompt（含子任务目标 + 原始问题）+ 历史消息 + 工具声明交给 LLM；
- LLM 返回 `tool_calls` → 执行；返回纯文本 → 视为最终答复；
- `react_observe`：扫描最近 assistant 消息判定是否完成；`max_react_steps` 防死循环。

### 3.3 Reflection 自我纠错（core/graph/reflector.py）
- 子任务"完成"后，Reflector 让 LLM 复核 `result` 是否正确充分；
- verdict=retry 时注入反馈、重置 react_steps、回到 react_reason 重做；
- `max_reflections`（默认 2）限制重试次数，避免无限纠错。

### 3.4 长程任务规划（core/graph/planner.py）
- Planner 把复杂问题拆为 ≤ `max_subtasks` 个有序子任务（JSON 输出，带 fence 兼容）；
- 逐个完成后 Aggregator 汇总各子任务结果生成最终答案；
- 解析失败降级为单任务，保证鲁棒。

### 3.5 多工具并发（core/graph/executor.py + concurrency.py）
- 同一轮 LLM 产出多个 `tool_calls` 时，用 `ThreadPoolExecutor` 并发执行；
- 每个工具调用独立 Tracer span，结果按 `tool_call_id` 回填 messages。

## 4. MCP 风格工具注册（core/tools）

与 MCP tool 定义对齐：

```python
Tool = {
    "name": str,
    "description": str,
    "input_schema": JSONSchema,   # 参数 schema
    "handler": callable
}
```

- `@register_tool` 装饰 `Tool` 子类即自动注册；
- `@register_function` 装饰普通函数也可（包装为 FunctionTool）；
- `as_functions()` 转 OpenAI/智谱 function_call 格式交给 LLM；
- `execute(name, args, tracer)` 统一调用入口，带 span 与错误兜底。

内置工具：
| 工具 | 说明 | 安全 |
|---|---|---|
| `knowledge_search` | RAG 检索 | — |
| `web_search` | 互联网搜索，无 API 时 stub 兜底 | 超时控制 |
| `python_exec` | 子进程执行 Python | 黑名单 + 超时 + 独立进程 |

**新增工具零侵入**：写一个 Tool 子类 + 装饰，不改 Agent 主干。

## 5. RAG 知识库（rag/）

流水线四步：

```
文档 ──chunking──▶ chunks ──embed──▶ vectors ──FAISS──▶ Top-K ──rerank──▶ Top-N
```

| 环节 | 实现要点 |
|---|---|
| 切分 | 递归字符切分（段落→换行→句号→空格），chunk_size/overlap 可配 |
| Embedding | 智谱 embedding-3（OpenAI 兼容），**按文本 hash 本地缓存**避免重复计费；无 Key 回退确定性 hash 向量 |
| 向量检索 | FAISS `IndexFlatIP`，L2 归一化后内积 = cosine，召回 Top-K（默认 20） |
| 重排 | API reranker 优先；**无 API 回退 MMR**（最大边际相关性，兼顾相关性与多样性），取 Top-N（默认 5） |

关注点：延迟（缓存 + 并发工具）、成本（token 归因定位热点）、可观测（检索/重排均进 Tracer span）。

## 6. 可观测性（observability/）

### Tracer
- 每个节点/工具/LLM 调用一条 span，含父子层级、起止时间、属性、状态；
- `tracer.tree()` 渲染缩进树，`to_json()` 可导出对接外部系统。

### CostTracker
- 从 LLM 响应 `usage` 累计 prompt/completion/total tokens 与费用；
- 按 scope（node/tool）归因，run 末输出分维度报告；
- 价格表 `PRICE_CNY_PER_1K` 可按网关实际计费调整。

## 7. LLM 接入：智谱 GLM coding-plan 网关（core/llm）

- `LLMClient` 抽象统一 `chat` / `chat_with_tools`；
- `GLMClient` 用 openai SDK，`base_url` 指向 coding-plan 网关（OpenAI 兼容）；
- 配置优先复用环境变量：`ZCODE_BASE_URL` / `ZAI_BUSINESS_BASE_URL` / `ZHIPU_API_KEY`；
- 适配智谱 `usage` 字段（含 prompt/completion/total）。

---

## 8. 自研框架 vs LangChain 取舍

| 维度 | 自研框架 | LangChain |
|---|---|---|
| **编排** | LangGraph 显式图，节点/条件边清晰可控，状态变更必须经节点返回 | Agent 抽象多、回调链隐式，调试链路长 |
| **工具** | MCP 风格极简 schema，注册即用 | Tool/StructuredTool/BaseTool 层级重 |
| **LLM 耦合** | 单一 LLMClient 抽象，换 Provider 改一个类 | 适配多但版本碎片化 |
| **可观测** | 内建 span + token 成本，零额外依赖 | 强依赖 LangSmith / callbacks 生态 |
| **生态** | loader/splitter 等需自建 | 现成组件海量 |
| **依赖体积** | 轻（无 torch） | 重 |

**定位**：为可控性、可观测性、轻依赖而自研；用 **LangGraph 做编排而非 LangChain Agent**。
在需要快速堆集成的场景仍可按需引入 LangChain 的个别组件（如 document loader）。

---

## 9. 目录结构

```
agent-project/
├── DESIGN.md / README.md / requirements.txt / .env.example / config.py / main.py
├── core/
│   ├── llm/      (base / glm provider)
│   ├── tools/    (base / registry / builtin: retrieval, web_search, code_exec)
│   ├── graph/    (state / planner / react / executor / reflector / builder)
│   └── concurrency.py
├── rag/          (chunking / embedder / vector_store / reranker / pipeline)
├── observability/ (tracer / cost_tracker)
├── data/knowledge/ (示例文档)
└── tests/        (mock，免 Key 跑通)
```

## 10. 扩展点

- 新工具：`@register_tool` 写 Tool 子类；
- 新 LLM：实现 `LLMClient`，在 main.py 替换 GLMClient；
- 新 Embedder/Reranker：实现 `Embedder` / `Reranker` 抽象；
- 接 OTel：`Tracer.to_json()` 已是 span 列表，加一层 OTel exporter 即可。
