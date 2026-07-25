"""SQLAlchemy ORM 模型定义。

本模块定义了系统中所有数据库表的 ORM 模型：
- User / UserSession: 用户认证相关（已在 0001 迁移中创建）
- Conversation / Message / QATurn: 多轮诊断对话相关（在 0002 迁移中创建）

表关系：
    User 1:N Conversation 1:N Message
                        1:N QATurn → references Message (user_message_id, assistant_message_id)
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.database import Base


class User(Base):
    """用户账号表。存储用户名、密码哈希、管理员标志。"""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)  # Argon2 哈希
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    sessions: Mapped[list["UserSession"]] = relationship(back_populates="user")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user")


class UserSession(Base):
    """登录会话表。存储 session token 的 SHA-256 哈希和过期时间。"""
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    session_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)  # cookie token 的 SHA-256
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 主动登出时设置

    user: Mapped[User] = relationship(back_populates="sessions")


class Conversation(Base):
    """对话会话表。存储诊断会话的元数据和当前诊断状态。

    字段说明：
        user_id: 所属用户 ID
        title: 会话标题（首条消息截断 50 字符）
        phase: 诊断阶段（collecting / clarifying / answered）
        diagnostic_state: JSONB，存储 DiagnosticState 的序列化数据
            包含 turn、phase、max_turns、symptoms、all_entity_names、topic_id
    """
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    phase: Mapped[str] = mapped_column(String(20), nullable=False, default="collecting")
    diagnostic_state: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        order_by=lambda: (Message.created_at, Message.id),
    )
    qa_turns: Mapped[list["QATurn"]] = relationship(back_populates="conversation", order_by="QATurn.created_at")


class Message(Base):
    """对话消息表。存储纯文本的用户/助手消息。

    只存对话内容，不混入 RAG 系统数据（那些存在 QATurn 中）。
    role 为 "user" 或 "assistant"。
    """
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class QATurn(Base):
    """QA 轮次表。存储每轮 RAG 检索的快照数据。

    与 Message 的职责分离：
    - Message 存用户/助手的对话文本（给用户看的）
    - QATurn 存系统内部的 RAG 数据（给开发者调试、给后续轮次用的）

    字段说明：
        user_message_id: 本轮用户消息的 ID（FK → messages.id）
        assistant_message_id: 本轮助手消息的 ID（FK → messages.id，生成完毕后回填）
        mode: "initial"（首轮 RAG）或 "followup"（后续轮次重评分）
        turn_number: 轮次序号（从 1 开始）
        keywords_json: 本轮提取的关键词 {"local": [...], "global": [...]}
        candidate_pool_json: 本轮候选实体池（完整列表，含评分、属性、邻居、来源）
            后续轮次通过 retrieve_followup() 加载此数据做证据文本重评分
        prompt_entities_json: 实际送入 prompt 的实体子集（精简版，仅名称和评分）
        memory_state_json: 本轮更新后的 DiagnosticState 快照
    """
    __tablename__ = "qa_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), nullable=False, index=True)
    user_message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), nullable=False)
    assistant_message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"), nullable=True)
    mode: Mapped[str] = mapped_column(String(20), nullable=False)
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    keywords_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    candidate_pool_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    prompt_entities_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    memory_state_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="qa_turns")
