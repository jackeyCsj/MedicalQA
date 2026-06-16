from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db.database import init_db
from backend.routers.auth import router as auth_router
from backend.routers.chat import router as chat_router
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # 预加载 QA 服务（向量索引 + embedding 模型），避免首次请求等待
    from backend.routers.chat import service
    service.load()
    yield


app = FastAPI(title="MedicalQA API", lifespan=lifespan)
app.add_middleware(
    # CORS 中间件 解决跨域问题——前端在 localhost:5173，后端在 localhost:8000，
    # 浏览器默认会拦截跨域请求，加了这个就放行了。
    CORSMiddleware, 
    allow_origins=[origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(chat_router)


"""
  python -m uvicorn backend.main:app --reload    # 后端
    http://127.0.0.1:8000
  cd frontend && npm.cmd run dev                  # 前端
    http://localhost:5173

"""
