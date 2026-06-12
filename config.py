"""项目集中配置。

所有外部依赖参数（Neo4j 连接、ollama 模型名、本地模型/数据路径）都通过环境变量
覆盖，未设置时使用与原代码完全一致的默认值，保证零回归。

使用方式::

    from config import settings
    print(settings.NEO4J_URL)

环境变量列表见下方常量定义；亦可参考 README 的「环境变量」章节。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    """读取环境变量，未设置时返回 default。"""
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔环境变量。"""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量，非法时返回 default。"""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """运行时配置容器（不可变）。"""

    # --- Neo4j ---
    NEO4J_URL: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str
    NEO4J_DBNAME: str

    # --- LLM（支持 ollama 本地 / 阿里百炼 DashScope 云端） ---
    LLM_PROVIDER: str           # "ollama" 或 "dashscope"
    DASHSCOPE_API_KEY: str      # 百炼 API Key（环境变量 DASHSCOPE_API_KEY）
    DASHSCOPE_MODEL: str        # 百炼模型 ID
    DASHSCOPE_BASE_URL: str     # 百炼兼容接口地址
    DASHSCOPE_MODELS: str       # 可选模型列表，逗号分隔

    # --- Ollama（向后保留，LLM_PROVIDER="ollama" 时使用） ---
    OLLAMA_QWEN_MODEL: str
    OLLAMA_LLAMA_MODEL: str

    # --- 本地模型与数据路径 ---
    NER_MODEL_NAME: str         # HuggingFace BERT 路径（chinese-roberta-wwm-ext）
    NER_CHECKPOINT: str         # 训练好的 NER 权重文件名（不含 .pt 后缀）
    DATA_DIR: str
    TMP_DIR: str
    MODEL_DIR: str

    # --- 日志 ---
    LOG_LEVEL: str

    # --- FastAPI / Auth ---
    DATABASE_URL: str
    SESSION_SECRET: str
    SESSION_COOKIE_NAME: str
    SESSION_EXPIRE_DAYS: int
    COOKIE_SECURE: bool
    CORS_ORIGINS: str


def _build_settings() -> Settings:
    return Settings(
        NEO4J_URL=_env("NEO4J_URL", "bolt://localhost:7687"),
        NEO4J_USER=_env("NEO4J_USER", "neo4j"),
        NEO4J_PASSWORD=_env("NEO4J_PASSWORD", "csj123456"),
        NEO4J_DBNAME=_env("NEO4J_DBNAME", "neo4j"),
        LLM_PROVIDER=_env("LLM_PROVIDER", "dashscope"),
        DASHSCOPE_API_KEY=_env("DASHSCOPE_API_KEY", ""),
        DASHSCOPE_MODEL=_env("DASHSCOPE_MODEL", "qwen-plus-latest"),
        DASHSCOPE_BASE_URL=_env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        DASHSCOPE_MODELS=_env("DASHSCOPE_MODELS", "qwen3.7-plus,qwen3.7-max,deepseek-v4-flash,deepseek-v4-pro,kimi-k2.6,qwen3.6-flash"),
        OLLAMA_QWEN_MODEL=_env("OLLAMA_QWEN_MODEL", "qwen:32b"),
        OLLAMA_LLAMA_MODEL=_env("OLLAMA_LLAMA_MODEL", "llama2-chinese:13b-chat-q8_0"),
        NER_MODEL_NAME=_env("NER_MODEL_NAME", "model/chinese-roberta-wwm-ext"),
        NER_CHECKPOINT=_env("NER_CHECKPOINT", "best_roberta_rnn_model_ent_aug"),
        DATA_DIR=_env("DATA_DIR", "data"),
        TMP_DIR=_env("TMP_DIR", "tmp_data"),
        MODEL_DIR=_env("MODEL_DIR", "model"),
        LOG_LEVEL=_env("LOG_LEVEL", "INFO"),
        DATABASE_URL=_env(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/medicalqa",
        ),
        SESSION_SECRET=_env("SESSION_SECRET", "dev-insecure-change-me"),
        SESSION_COOKIE_NAME=_env("SESSION_COOKIE_NAME", "session"),
        SESSION_EXPIRE_DAYS=_env_int("SESSION_EXPIRE_DAYS", 7),
        COOKIE_SECURE=_env_bool("COOKIE_SECURE", False),
        CORS_ORIGINS=_env("CORS_ORIGINS", "http://localhost:5173"),
    )


# 全局单例：模块导入时即冻结一份当前环境变量快照
settings: Settings = _build_settings()
