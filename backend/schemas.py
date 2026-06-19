"""Pydantic 请求/响应模型。

定义 FastAPI 端点的请求体和响应体的 schema，FastAPI 自动做：
- 请求体 JSON → 模型实例的反序列化
- 模型实例 → 响应体 JSON 的序列化
- 字段验证（min_length、max_length、ge、le 等）
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AuthRequest(BaseModel):
    """认证请求（注册/登录）。用户名 3-64 字符，密码 8-128 字符。"""
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    """用户信息响应。"""
    username: str
    is_admin: bool


class LoginResponse(BaseModel):
    """登录成功响应。"""
    user: UserResponse


class OkResponse(BaseModel):
    """通用成功响应。"""
    ok: bool


class ChatRequest(BaseModel):
    """聊天请求体。

    字段说明：
        message: 用户消息文本（1-1000 字符）
        model: LLM 模型名称（None 时使用默认模型）
        final_entity_top_k: 送入 prompt 的实体数量（3-10），影响回答的信息量
        conversation_id: 会话 ID（None 表示新会话，有值表示继续已有会话）
    """
    message: str = Field(min_length=1, max_length=1000)
    model: str | None = None
    final_entity_top_k: int = Field(default=5, ge=3, le=10)
    conversation_id: int | None = None


class ModelsResponse(BaseModel):
    """模型列表响应。"""
    provider: str
    models: list[str]


class StreamDebug(BaseModel):
    """调试信息（SSE debug 事件的数据结构）。"""
    entities: dict[str, str] | None = None
    retrieval: dict[str, Any] | None = None
    intents: str | None = None
    knowledge: str | None = None
