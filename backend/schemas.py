from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    username: str
    is_admin: bool


class LoginResponse(BaseModel):
    user: UserResponse


class OkResponse(BaseModel):
    ok: bool


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    model: str | None = None
    final_entity_top_k: int = Field(default=5, ge=3, le=10)


class ModelsResponse(BaseModel):
    provider: str
    models: list[str]


class StreamDebug(BaseModel):
    entities: dict[str, str] | None = None
    retrieval: dict[str, Any] | None = None
    intents: str | None = None
    knowledge: str | None = None
