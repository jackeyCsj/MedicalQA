from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.routers.auth import current_user
from backend.schemas import ChatRequest, ModelsResponse
from backend.services.qa_service import MedicalQAService
from config import settings


router = APIRouter(prefix="/api", tags=["chat"])
service = MedicalQAService()


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/models", response_model=ModelsResponse)
def models():
    return ModelsResponse(provider=settings.LLM_PROVIDER, models=service.get_models())


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest, user=Depends(current_user)):
    async def event_gen():
        try:
            async for event, data in service.stream_answer(payload.message, payload.model or ""):
                yield _sse(event, data)
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")
