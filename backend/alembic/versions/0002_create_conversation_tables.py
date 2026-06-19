"""创建多轮诊断对话的三张核心表：conversations、messages、qa_turns。

表设计思路：
- conversations: 存储对话会话元数据和当前诊断状态（JSONB）
- messages: 纯对话消息文本（user/assistant），给用户看的
- qa_turns: 每轮 RAG 检索快照（关键词、候选池、prompt 实体），系统内部数据

职责分离：messages 存对话记录，qa_turns 存系统计算数据，不混在同一张表。

Revision ID: 0002_create_conversation_tables
Revises: 0001_create_auth_tables
Create Date: 2026-06-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# Alembic 版本标识：0002，依赖 0001（auth 表）
revision = "0002_create_conversation_tables"
down_revision = "0001_create_auth_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- conversations 表：对话会话 ---
    # 存储每个诊断会话的元数据，diagnostic_state 用 JSONB 存诊断状态机的当前状态
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False, server_default=""),
        sa.Column("phase", sa.String(length=20), nullable=False, server_default="collecting"),
        sa.Column("diagnostic_state", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_conversations_id", "conversations", ["id"])
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])

    # --- messages 表：对话消息 ---
    # 纯文本消息，role 为 "user" 或 "assistant"，不包含任何 RAG 系统数据
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("conversation_id", sa.Integer(), sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("role", sa.String(length=10), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_messages_id", "messages", ["id"])
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])

    # --- qa_turns 表：每轮 RAG 检索快照 ---
    # 存储每轮诊断的系统内部数据，供调试和后续轮次重评分使用
    # candidate_pool_json 存完整候选实体池，retrieve_followup() 从这里加载上轮候选
    op.create_table(
        "qa_turns",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("conversation_id", sa.Integer(), sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("user_message_id", sa.Integer(), sa.ForeignKey("messages.id"), nullable=False),
        sa.Column("assistant_message_id", sa.Integer(), sa.ForeignKey("messages.id"), nullable=True),
        sa.Column("mode", sa.String(length=20), nullable=False),
        sa.Column("turn_number", sa.Integer(), nullable=False),
        sa.Column("keywords_json", JSONB(), nullable=True),
        sa.Column("candidate_pool_json", JSONB(), nullable=True),
        sa.Column("prompt_entities_json", JSONB(), nullable=True),
        sa.Column("memory_state_json", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_qa_turns_id", "qa_turns", ["id"])
    op.create_index("ix_qa_turns_conversation_id", "qa_turns", ["conversation_id"])


def downgrade() -> None:
    """回滚：按外键依赖的逆序删除表和索引。"""
    op.drop_index("ix_qa_turns_conversation_id", table_name="qa_turns")
    op.drop_index("ix_qa_turns_id", table_name="qa_turns")
    op.drop_table("qa_turns")
    op.drop_index("ix_messages_conversation_id", table_name="messages")
    op.drop_index("ix_messages_id", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_index("ix_conversations_id", table_name="conversations")
    op.drop_table("conversations")
