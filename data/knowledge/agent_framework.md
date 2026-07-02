# 自研 Agent 调度框架

本框架基于 Python 与 LangGraph 实现，核心能力包括：

## ReAct 推理
Agent 在每一步先思考（Thought）再决定动作（Action），观察结果（Observation）后进入下一轮，
循环直至能给出最终答案。本实现通过 LLM 的 Function Calling 能力驱动工具调用，
tool_calls 为空即视为 Agent 已得出结论。

## Reflection 自我纠错
每个子任务产出结果后，由独立的 Reflector 节点复核：若结果不充分或有误，
注入反馈并回到推理节点重做，最多重试 max_reflections 次（默认 2 次）。

## 长程任务规划
面对复杂问题，Planner 节点先将其拆解为有序子任务列表，
Agent 逐个完成子任务后由 Aggregator 汇总，避免单次推理跨度太大。

## 多工具并发编排
同一轮 ReAct 中若 LLM 产出多个 tool_calls，tool_executor 用线程池并发执行，
降低串行等待延迟。

## MCP 风格工具注册
工具定义为 {name, description, input_schema, handler}，与 MCP tool 定义对齐。
@register_tool 装饰器自动注册，as_functions() 转 OpenAI/智谱 function_call 格式。
新增工具零侵入，无需改动 Agent 主干。
