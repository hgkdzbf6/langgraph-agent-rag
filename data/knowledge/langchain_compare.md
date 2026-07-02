# 自研框架与 LangChain 的取舍

## 为什么自研
- 流程可控：LangGraph 显式声明节点与条件边，ReAct 循环、Reflection 回退、子任务推进
  都在图拓扑里一目了然，便于调试与改造；LangChain Agent 的回调链较隐式。
- 工具抽象轻：MCP 风格的 Tool {name, description, input_schema, handler} 极简，
  注册即用；LangChain 的 Tool/StructuredTool/BaseTool 层级更重。
- LLM 解耦：单一 LLMClient 抽象，换 Provider 只改一个类，不牵连业务节点。
- 内建可观测：Tracer span + CostTracker token 成本归因零额外依赖；
  LangChain 强依赖 LangSmith / callbacks 生态。
- 轻依赖：核心只依赖 langgraph + openai + faiss + numpy，不引入 torch。

## 自研的代价
- 生态组件少：文档 loader、splitter、verifier 等需自建；
- 成熟度：没有 LangChain 社区沉淀的数百个集成；
- 适合场景：对可控性、可观测性、轻量化有要求的内部平台。

## 结论
定位是「为可控性、可观测性、轻依赖而自研，用 LangGraph 做编排而非 LangChain Agent」，
在需要快速堆集成的场景仍可按需引入 LangChain 的个别组件（如 document loader）。
