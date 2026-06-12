from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.schemas import AuthRequest, LoginResponse, OkResponse, UserResponse
from backend.services.auth_service import (
    AuthError,
    AuthRateLimitError,
    authenticate_user,
    create_session,
    create_user,
    get_user_from_token,
    revoke_session,
)
from config import settings


router = APIRouter(prefix="/api/auth", tags=["auth"])


def _cookie_settings() -> dict[str, object]:
    return {
        "httponly": True,
        "samesite": "lax",
        "secure": settings.COOKIE_SECURE,
        "path": "/",
        "max_age": settings.SESSION_EXPIRE_DAYS * 24 * 3600,
    }


def _get_cookie_token(request: Request) -> str | None:
    return request.cookies.get(settings.SESSION_COOKIE_NAME)


def current_user(request: Request, db: Session = Depends(get_db)):
    token = _get_cookie_token(request)
    user = get_user_from_token(db, token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录")
    return user


@router.post("/register", response_model=OkResponse)
def register(payload: AuthRequest, db: Session = Depends(get_db)):
    try:
        create_user(db, payload.username, payload.password, is_admin=False)
    except AuthError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="注册失败，请更换用户名或密码")
    return OkResponse(ok=True)


@router.post("/login", response_model=LoginResponse)
def login(payload: AuthRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    client_host = request.client.host if request.client else ""
    try:
        user = authenticate_user(db, payload.username, payload.password, client_host=client_host)
    except AuthRateLimitError:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="登录尝试过于频繁，请稍后再试")
    except AuthError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户名或密码错误")

    token, _ = create_session(db, user)
    response.set_cookie(settings.SESSION_COOKIE_NAME, token, **_cookie_settings())
    return LoginResponse(user=UserResponse(username=user.username, is_admin=user.is_admin))


@router.post("/logout", response_model=OkResponse)
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    token = _get_cookie_token(request)
    if token:
        revoke_session(db, token)
    response.delete_cookie(settings.SESSION_COOKIE_NAME, path="/")
    return OkResponse(ok=True)


@router.get("/me", response_model=UserResponse)
def me(user=Depends(current_user)):
    return UserResponse(username=user.username, is_admin=user.is_admin)
