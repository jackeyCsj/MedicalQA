"""项目集中配置。

所有配置项通过环境变量读取，默认值统一定义在 .env.example 中。
使用前请先复制模板：cp .env.example .env

使用方式::

    from config import settings
    print(settings.NEO4J_URL)

环境变量列表见 .env.example；亦可参考 README 的「环境变量」章节。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 自动加载项目根目录下的 .env 文件（不覆盖已有环境变量）
load_dotenv(Path(__file__).resolve().parent / ".env")


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

    # --- LLM 通用配置（支持 dashscope / mimo / deepseek / ollama） ---
    #   优先级：LLM_* > 厂商专属环境变量（DASHSCOPE_API_KEY 等）
    #   所有默认值定义在 .env.example 中
    LLM_PROVIDER: str           # "dashscope" / "mimo" / "deepseek" / "ollama"
    LLM_API_KEY: str            # 当前生效的 API Key
    LLM_BASE_URL: str           # 当前生效的 OpenAI 兼容接口地址
    LLM_MODEL: str              # 当前生效的默认模型
    LLM_MODELS: str             # 可选模型列表，逗号分隔

    # --- Ollama（LLM_PROVIDER="ollama" 时使用） ---
    #   Ollama 走 ollama Python 库而非 OpenAI 兼容 API，模型名格式不同（qwen:32b）
    #   因此单独配置，不走 LLM_MODEL 通用链路
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

    # --- 多轮诊断 ---
    #   MAX_CLARIFICATION_TURNS: 最大追问轮次（默认 2，建议 2-3）
    #     达到此轮次后，decide_action() 强制进入 answer 阶段
    #   CLARIFICATION_ENABLED: 是否启用多轮诊断
    #     False 时退回单轮模式（不走路由，不存 DB，向后兼容）
    MAX_CLARIFICATION_TURNS: int
    CLARIFICATION_ENABLED: bool


def _build_settings() -> Settings:
    """从环境变量读取全部配置，所有默认值定义在 .env.example 中。"""
    provider = _env("LLM_PROVIDER", "")
    pfx = provider.upper()  # DASHSCOPE / MIMO / DEEPSEEK / OLLAMA

    # 优先级：LLM_* > 厂商专属环境变量（DASHSCOPE_API_KEY 等）
    llm_api_key = _env("LLM_API_KEY", _env(f"{pfx}_API_KEY", ""))
    llm_base_url = _env("LLM_BASE_URL", _env(f"{pfx}_BASE_URL", ""))
    llm_model = _env("LLM_MODEL", _env(f"{pfx}_MODEL", ""))
    llm_models = _env("LLM_MODELS", _env(f"{pfx}_MODELS", ""))

    return Settings(
        NEO4J_URL=_env("NEO4J_URL", ""),
        NEO4J_USER=_env("NEO4J_USER", ""),
        NEO4J_PASSWORD=_env("NEO4J_PASSWORD", ""),
        NEO4J_DBNAME=_env("NEO4J_DBNAME", ""),
        LLM_PROVIDER=provider,
        LLM_API_KEY=llm_api_key,
        LLM_BASE_URL=llm_base_url,
        LLM_MODEL=llm_model,
        LLM_MODELS=llm_models,
        OLLAMA_QWEN_MODEL=_env("OLLAMA_QWEN_MODEL", ""),
        OLLAMA_LLAMA_MODEL=_env("OLLAMA_LLAMA_MODEL", ""),
        NER_MODEL_NAME=_env("NER_MODEL_NAME", ""),
        NER_CHECKPOINT=_env("NER_CHECKPOINT", ""),
        DATA_DIR=_env("DATA_DIR", ""),
        TMP_DIR=_env("TMP_DIR", ""),
        MODEL_DIR=_env("MODEL_DIR", ""),
        LOG_LEVEL=_env("LOG_LEVEL", ""),
        DATABASE_URL=_env("DATABASE_URL", ""),
        SESSION_SECRET=_env("SESSION_SECRET", ""),
        SESSION_COOKIE_NAME=_env("SESSION_COOKIE_NAME", ""),
        SESSION_EXPIRE_DAYS=_env_int("SESSION_EXPIRE_DAYS", 0),
        COOKIE_SECURE=_env_bool("COOKIE_SECURE", False),
        CORS_ORIGINS=_env("CORS_ORIGINS", ""),
        MAX_CLARIFICATION_TURNS=_env_int("MAX_CLARIFICATION_TURNS", 0),
        CLARIFICATION_ENABLED=_env_bool("CLARIFICATION_ENABLED", True),
    )


# 全局单例：模块导入时即冻结一份当前环境变量快照
settings: Settings = _build_settings()



if __name__ == "__main__":
    # 直接运行 config.py 可打印当前配置（测试环境变量加载）
    import json
    print(json.dumps(settings.__dict__, indent=2, ensure_ascii=False))