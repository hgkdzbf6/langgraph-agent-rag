# RAG 知识库

检索增强生成（RAG）流水线分四步：

## 1. 文档切分
采用递归字符切分：按段落 → 换行 → 句号 → 空格层级递归，
在目标 chunk_size 内尽量保持语义完整，并加入 chunk_overlap 重叠以保留上下文连续性。

## 2. Embedding
调用智谱 embedding-3 接口（OpenAI 兼容），对每个 chunk 生成稠密向量，
并按文本 hash 本地缓存，避免重复计费。无 Key 时回退确定性 hash 向量，保证流程不中断。

## 3. 向量检索
使用 FAISS IndexFlatIP。向量先做 L2 归一化，归一化后内积即等于 cosine 相似度，
召回 Top-K（默认 20）候选。

## 4. 重排序
对召回候选用 Reranker 重排取 Top-N（默认 5）：
- 优先调用兼容 rerank API；
- 无 API 时回退 MMR（最大边际相关性），兼顾相关性与多样性，避免结果同质化。

## 关注点
- 延迟：embedding 缓存 + 并发工具调用降低端到端时延；
- 成本：token 统计按节点/工具归因，便于定位成本热点；
- 可观测性：RAG 检索/重排均纳入 Tracer span，输出链路树。
