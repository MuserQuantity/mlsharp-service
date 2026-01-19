from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from app.core.config import settings


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


ApiKeyDep = Depends(require_api_key)
