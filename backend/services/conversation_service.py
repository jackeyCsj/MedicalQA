"""会话服务：数据库 CRUD 操作。

本模块封装了多轮诊断对话中所有与数据库交互的操作，包括：
- 会话（Conversation）的创建、查询、列表、删除
- 消息（Message）的保存和查询
- QA 轮次（QATurn）的创建、更新、查询

设计原则：
- 所有函数接收 db: Session 作为参数（由 FastAPI 的 Depends 注入）
- 使用 db.flush() 而非 db.commit()，事务由调用方（FastAPI 请求生命周期）统一提交
- 函数保持无状态，不依赖模块级变量
"""

from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from backend.db.models import Conversation, Message, QATurn

logger = logging.getLogger(__name__)


# ============================================================
#  会话（Conversation）操作
# ============================================================

def create_conversation(db: Session, user_id: int, title: str, diagnostic_state: dict) -> Conversation:
    """创建新的诊断会话。

    在多轮诊断的首轮（conversation_id=None）时调用。
    title 通常取用户首条消息的前 50 个字符。
    diagnostic_state 存储初始诊断状态（DiagnosticState.to_dict()）。

    参数：
        db: SQLAlchemy 会话
        user_id: 当前登录用户的 ID
        title: 会话标题（显示在前端侧边栏）
        diagnostic_state: 初始诊断状态的字典形式
    """
    conv = Conversation(
        user_id=user_id,
        title=title,
        phase="collecting",
        diagnostic_state=diagnostic_state,
    )
    db.add(conv)
    db.flush()  # flush 获得自增 ID，但不提交事务
    logger.info("[db] 创建会话 id=%d, title='%s'", conv.id, title)
    return conv


def get_conversation(db: Session, conv_id: int, user_id: int) -> Optional[Conversation]:
    """按 ID 查询会话，同时验证归属权（user_id 匹配）。

    用于后续轮次加载诊断状态，以及 API 端点的权限校验。
    """
    return db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == user_id,
    ).first()


def list_conversations(db: Session, user_id: int, limit: int = 50) -> List[Conversation]:
    """查询用户的会话列表，按更新时间倒序排列。

    用于前端侧边栏的历史会话展示。limit 默认 50，防止一次加载过多。
    """
    return db.query(Conversation).filter(
        Conversation.user_id == user_id,
    ).order_by(Conversation.updated_at.desc()).limit(limit).all()


def delete_conversation(db: Session, conv_id: int, user_id: int) -> bool:
    """删除会话及其所有关联数据（消息 + QA 轮次）。

    按外键依赖顺序删除：先删 qa_turns，再删 messages，最后删 conversation。
    返回 True 表示删除成功，False 表示会话不存在或无权访问。
    """
    conv = get_conversation(db, conv_id, user_id)
    if not conv:
        return False
    # 先删子表（qa_turns 引用 messages，所以先删 qa_turns）
    db.query(QATurn).filter(QATurn.conversation_id == conv_id).delete()
    db.query(Message).filter(Message.conversation_id == conv_id).delete()
    db.delete(conv)
    db.flush()
    logger.info("[db] 删除会话 id=%d", conv_id)
    return True


def update_conversation_phase(db: Session, conv: Conversation, phase: str) -> None:
    """更新会话的诊断阶段（collecting / clarifying / answered）。

    phase 变化路径：collecting → clarifying → answered
    前端侧边栏会根据 phase 显示"追问中"徽章。
    """
    conv.phase = phase
    db.flush()


def update_diagnostic_state(db: Session, conv: Conversation, state_dict: dict) -> None:
    """更新会话的诊断状态（JSONB 字段）。

    每轮对话结束后调用，将最新的 DiagnosticState 序列化后写入 DB。
    下一轮对话开始时从 DB 加载恢复状态。
    """
    conv.diagnostic_state = state_dict
    db.flush()


# ============================================================
#  消息（Message）操作
# ============================================================

def save_message(db: Session, conversation_id: int, role: str, content: str) -> Message:
    """保存一条对话消息。

    role 为 "user" 或 "assistant"。content 为纯文本消息内容。
    消息只存对话文本，不混入 RAG 系统数据（那些存在 qa_turns 中）。

    返回保存后的 Message 对象（含自增 ID），供 qa_turns 引用。
    """
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
    )
    db.add(msg)
    db.flush()
    return msg


def get_conversation_messages(db: Session, conversation_id: int) -> List[Message]:
    """查询会话的全部消息，按创建时间正序排列。

    用于：1) 前端展示对话历史  2) 构建 prompt 中的对话历史上下文
    """
    return db.query(Message).filter(
        Message.conversation_id == conversation_id,
    ).order_by(Message.created_at).all()


# ============================================================
#  QA 轮次（QATurn）操作
# ============================================================

def create_qa_turn(
    db: Session,
    conversation_id: int,
    user_message_id: int,
    mode: str,
    turn_number: int,
    keywords_json: dict,
    candidate_pool_json: list,
    prompt_entities_json: list,
    memory_state_json: dict,
) -> QATurn:
    """创建一个 QA 轮次记录，保存本轮的 RAG 检索快照。

    QA 轮次是系统内部数据（给开发者调试、给后续轮次用的），与用户可见的 messages 分离。

    参数：
        conversation_id: 所属会话 ID
        user_message_id: 本轮用户消息的 ID
        mode: "initial"（首轮 RAG 检索）或 "followup"（后续轮次重评分）
        turn_number: 轮次序号（从 1 开始）
        keywords_json: 本轮提取的关键词 {"local": [...], "global": [...]}
        candidate_pool_json: 本轮候选实体池（完整列表，含评分、属性、邻居）
        prompt_entities_json: 实际送入 prompt 的实体子集（精简版）
        memory_state_json: 本轮更新后的诊断状态快照

    注意：assistant_message_id 初始为 None，等 LLM 生成完毕后通过 update_qa_turn_assistant() 回填。
    """
    turn = QATurn(
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        mode=mode,
        turn_number=turn_number,
        keywords_json=keywords_json,
        candidate_pool_json=candidate_pool_json,
        prompt_entities_json=prompt_entities_json,
        memory_state_json=memory_state_json,
    )
    db.add(turn)
    db.flush()
    return turn


def update_qa_turn_assistant(db: Session, turn: QATurn, assistant_message_id: int) -> None:
    """回填 QA 轮次的 assistant_message_id。

    LLM 流式生成完毕后调用，将助手消息 ID 关联到对应的 QA 轮次。
    这样可以通过 qa_turn 同时找到用户消息和助手消息。
    """
    turn.assistant_message_id = assistant_message_id
    db.flush()


def get_latest_qa_turn(db: Session, conversation_id: int) -> Optional[QATurn]:
    """获取会话中最新的 QA 轮次（按 turn_number 倒序取第一条）。

    用于后续轮次加载上轮的候选实体池（candidate_pool_json），
    供 retrieve_followup() 做证据文本重评分。
    """
    return db.query(QATurn).filter(
        QATurn.conversation_id == conversation_id,
    ).order_by(QATurn.turn_number.desc()).first()
