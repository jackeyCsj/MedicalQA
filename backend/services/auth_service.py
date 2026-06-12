from __future__ import annotations

import hashlib
import secrets
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Tuple

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from sqlalchemy.orm import Session

from backend.db.models import User, UserSession
from config import settings


ph = PasswordHasher()
_login_failures: Dict[Tuple[str, str], Deque[datetime]] = defaultdict(deque)
_failure_lock = threading.Lock()


class AuthError(Exception):
    """Base auth error."""


class AuthRateLimitError(AuthError):
    """Raised when repeated logins are throttled."""


def hash_password(password: str) -> str:
    return ph.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return ph.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def _session_token() -> str:
    return secrets.token_urlsafe(32)


def _session_hash(token: str) -> str:
    return hashlib.sha256(f"{settings.SESSION_SECRET}:{token}".encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _register_failure(username: str, client_host: str) -> None:
    key = (username.lower(), client_host or "unknown")
    cutoff = _utcnow() - timedelta(minutes=10)
    with _failure_lock:
        window = _login_failures[key]
        while window and window[0] < cutoff:
            window.popleft()
        window.append(_utcnow())


def _is_throttled(username: str, client_host: str) -> bool:
    key = (username.lower(), client_host or "unknown")
    cutoff = _utcnow() - timedelta(minutes=10)
    with _failure_lock:
        window = _login_failures[key]
        while window and window[0] < cutoff:
            window.popleft()
        return len(window) >= 5


def _clear_failures(username: str, client_host: str) -> None:
    key = (username.lower(), client_host or "unknown")
    with _failure_lock:
        _login_failures.pop(key, None)


def create_user(db: Session, username: str, password: str, is_admin: bool = False) -> User:
    if db.query(User).filter(User.username == username).first():
        raise AuthError("registration failed")
    user = User(username=username, password_hash=hash_password(password), is_admin=is_admin)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str, client_host: str = "") -> User:
    if _is_throttled(username, client_host):
        raise AuthRateLimitError("too many attempts")
    user = db.query(User).filter(User.username == username).first()
    if not user or user.disabled_at is not None:
        _register_failure(username, client_host)
        raise AuthError("bad credentials")
    if not verify_password(password, user.password_hash):
        _register_failure(username, client_host)
        raise AuthError("bad credentials")
    _clear_failures(username, client_host)
    return user


def create_session(db: Session, user: User) -> Tuple[str, UserSession]:
    token = _session_token()
    session = UserSession(
        user_id=user.id,
        session_hash=_session_hash(token),
        expires_at=_utcnow() + timedelta(days=settings.SESSION_EXPIRE_DAYS),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return token, session


def revoke_session(db: Session, token: str) -> None:
    session = db.query(UserSession).filter(UserSession.session_hash == _session_hash(token)).first()
    if session and session.revoked_at is None:
        session.revoked_at = _utcnow()
        db.commit()


def get_user_from_token(db: Session, token: str | None) -> User | None:
    if not token:
        return None
    session = (
        db.query(UserSession)
        .filter(UserSession.session_hash == _session_hash(token))
        .filter(UserSession.revoked_at.is_(None))
        .first()
    )
    if not session or _as_utc(session.expires_at) < _utcnow():
        return None
    user = session.user
    if not user or user.disabled_at is not None:
        return None
    return user
