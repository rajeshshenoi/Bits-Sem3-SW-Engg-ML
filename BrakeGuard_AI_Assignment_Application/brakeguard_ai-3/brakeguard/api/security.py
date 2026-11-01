"""API-key authentication dependency (model access control)."""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from ..logging_config import get_logger

logger = get_logger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(request: Request, api_key: str | None = Security(api_key_header)) -> None:
    """Reject requests without a valid ``X-API-Key`` when a key is configured.

    ``hmac.compare_digest`` gives a constant-time comparison, so the key
    cannot be guessed byte-by-byte through response-timing differences.
    """
    expected = request.app.state.container.settings.api_key
    if not expected:
        return
    if not api_key or not hmac.compare_digest(api_key, expected):
        logger.warning("Rejected request to %s: invalid or missing API key", request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
