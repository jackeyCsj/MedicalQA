"""聊天和会话管理的 API 端点。

本模块定义了以下端点：
- GET  /api/models: 获取可用 LLM 模型列表
- POST /api/chat/stream: SSE 流式问答（核心端点）
- GET  /api/conversations: 用户的会话列表
- GET  /api/conversations/{id}/messages: 会话消息历史
- DELETE /api/conversations/{id}: 删除会话

所有端点都需要登录认证（通过 current_user 依赖注入）。
"""

from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.routers.auth import current_user
from backend.schemas import ChatRequest, ModelsResponse
from backend.services import conversation_service as conv_svc
from backend.services.qa_service import MedicalQAService
from config import settings


router = APIRouter(prefix="/api", tags=["chat"])
service = MedicalQAService()  # 单例：懒加载 VectorIndex、KGClient、Retriever


def _sse(event: str, data) -> str:
    """将 SSE 事件格式化为标准格式：event: xxx\\ndata: json\\n\\n"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/models", response_model=ModelsResponse)
def models():
    """返回当前 LLM 提供商和可用模型列表。"""
    return ModelsResponse(provider=settings.LLM_PROVIDER, models=service.get_models())


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    user=Depends(current_user),
    db: Session = Depends(get_db),
):
    """SSE 流式问答端点（核心端点）。

    接收用户消息，通过 service.stream_answer() 执行完整的问答流程，
    以 SSE 事件流的形式返回结果。

    SSE 事件类型：
    - status: 进度提示（如"正在提取关键词..."）
    - debug: 调试信息（关键词、检索结果、prompt 等）
    - token: LLM 输出的文本片段
    - action: 路由决策结果 {"type": "ask"} 或 {"type": "answer"}
    - done: 完成，含 conversation_id 和 diagnostic_state
    - error: 错误信息
    """
    async def event_gen():
        try:
            async for event, data in service.stream_answer(
                payload.message,
                payload.model or "",
                payload.final_entity_top_k,
                conversation_id=payload.conversation_id,
                db=db,
                user_id=user.id,
            ):
                yield _sse(event, data)
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/conversations")
def list_conversations(user=Depends(current_user), db: Session = Depends(get_db)):
    """返回用户的会话列表（按更新时间倒序，最多 50 条）。

    用于前端侧边栏的历史会话展示。返回 id、title、phase、updated_at。
    """
    convs = conv_svc.list_conversations(db, user.id)
    return [
        {
            "id": c.id,
            "title": c.title,
            "phase": c.phase,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        }
        for c in convs
    ]


@router.get("/conversations/{conv_id}/messages")
def get_messages(conv_id: int, user=Depends(current_user), db: Session = Depends(get_db)):
    """返回会话的全部消息（按创建时间正序）。

    前端切换历史会话时调用，恢复对话内容。
    """
    conv = conv_svc.get_conversation(db, conv_id, user.id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = conv_svc.get_conversation_messages(db, conv_id)
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in messages
    ]


@router.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: int, user=Depends(current_user), db: Session = Depends(get_db)):
    """删除会话及其所有关联数据（消息 + QA 轮次）。"""
    ok = conv_svc.delete_conversation(db, conv_id, user.id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}
