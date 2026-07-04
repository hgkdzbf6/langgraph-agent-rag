"""全局配置加载。

优先复用环境已有变量（ZCODE_BASE_URL / ZAI_BUSINESS_BASE_URL），缺省指向智谱
coding-plan 网关（OpenAI 兼容）。复制 .env.example 为 .env 后填入真实 Key 即可。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    # override=True：.env 中显式配置优先于 shell 已有环境变量，
    # 便于针对本项目指向智谱 coding-plan 网关而非宿主默认值。
    load_dotenv(override=True)
except Exception:
    pass


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


@dataclass
class LLMConfig:
    base_url: str = field(
        default_factory=lambda: _env("ZCODE_BASE_URL")
        or _env("ZAI_BUSINESS_BASE_URL")
        or "https://open.bigmodel.cn/api/paas/v4"
    )
    api_key: str = field(
        default_factory=lambda: _env("ZHIPU_API_KEY") or _env("ZCODE_API_KEY")
    )
    model: str = field(default_factory=lambda: _env("GLM_MODEL", "GLM-5.2"))
    temperature: float = 0.3
    max_tokens: int = 2048
    # thinking 模式：disabled 关闭思维链（更快、无 reasoning_tokens）；
    # enabled 开启深度推理（更准、更慢）。默认 disabled 以提升响应速度。
    # 通过环境变量 ENABLE_THINKING=true 开启。
    thinking: str = field(default_factory=lambda: "enabled" if _env("ENABLE_THINKING", "").lower() in ("true", "1", "yes") else "disabled")

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)


@dataclass
class EmbedConfig:
    base_url: str = field(
        default_factory=lambda: _env("EMBED_BASE_URL")
        or _env("ZCODE_BASE_URL")
        or _env("ZAI_BUSINESS_BASE_URL")
        or "https://open.bigmodel.cn/api/paas/v4"
    )
    api_key: str = field(
        default_factory=lambda: _env("ZHIPU_API_KEY") or _env("ZCODE_API_KEY")
    )
    model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "embedding-3"))
    dim: int = 2048  # embedding-3 默认维度，按实际网关调整
    cache_dir: str = ".embed_cache"


@dataclass
class RerankConfig:
    base_url: str = field(
        default_factory=lambda: _env("ZCODE_BASE_URL")
        or _env("ZAI_BUSINESS_BASE_URL")
        or "https://open.bigmodel.cn/api/paas/v4"
    )
    api_key: str = field(
        default_factory=lambda: _env("ZHIPU_API_KEY") or _env("ZCODE_API_KEY")
    )
    model: str = field(default_factory=lambda: _env("RERANK_MODEL", "reranker"))


@dataclass
class RAGConfig:
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "1024")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "128")))
    retrieve_topk: int = field(default_factory=lambda: int(_env("RETRIEVE_TOPK", "20")))
    rerank_topn: int = field(default_factory=lambda: int(_env("RERANK_TOPN", "5")))
    index_path: str = "data/faiss.index"
    meta_path: str = "data/faiss.meta.json"
    knowledge_dir: str = field(default_factory=lambda: _env("KNOWLEDGE_DIR", "data/knowledge"))


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    # Agent 行为
    max_subtasks: int = 3
    max_react_steps: int = 3        # 单个子任务内 ReAct 最大轮次
    max_reflections: int = 2        # 单个步骤最多纠错次数
    enable_complexity_check: bool = True   # 启用复杂度前置判断
    adaptive_thinking: bool = True         # 按问题难度自动开关 thinking 思维链
    enable_reflection: bool = True        # 启用 Reflection 自我纠错
    enable_llm_cache: bool = False        # 启用 LLM 响应缓存
    sandbox_mode: str = "local"           # "local" | "docker"
    docker_image: str = "python:3.11-slim"
    docker_timeout: int = 10
    docker_memory: str = "128m"
    docker_cpus: float = 0.5


CONFIG = Config()
