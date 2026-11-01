"""Mapping from domain exceptions to HTTP responses."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..exceptions import (
    BrakeGuardError,
    FeatureEngineeringError,
    ModelNotReadyError,
    VehicleNotFoundError,
)
from ..logging_config import get_logger

logger = get_logger(__name__)

_STATUS_BY_EXCEPTION: list[tuple[type[BrakeGuardError], int, str]] = [
    (VehicleNotFoundError, status.HTTP_404_NOT_FOUND, "vehicle_not_found"),
    (ModelNotReadyError, status.HTTP_503_SERVICE_UNAVAILABLE, "model_not_ready"),
    (FeatureEngineeringError, status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_features"),
]


def error_response(request: Request, status_code: int, code: str, message: str, details=None) -> JSONResponse:
    """Build the uniform error envelope."""
    body = {
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", None),
            "details": details,
        }
    }
    return JSONResponse(status_code=status_code, content=jsonable_encoder(body))


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so every error uses :class:`ErrorResponse`."""

    @app.exception_handler(BrakeGuardError)
    async def _domain_error(request: Request, exc: BrakeGuardError):
        for exc_type, status_code, code in _STATUS_BY_EXCEPTION:
            if isinstance(exc, exc_type):
                logger.warning("%s %s -> %d %s", request.method, request.url.path, status_code, exc)
                return error_response(request, status_code, code, str(exc))
        logger.error("Unhandled domain error on %s: %s", request.url.path, exc)
        return error_response(request, 500, "internal_error", "Internal error")

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        details = [
            {"loc": list(err.get("loc", [])), "msg": err.get("msg"), "type": err.get("type")}
            for err in exc.errors()
        ]
        logger.warning("Rejected invalid request on %s: %d error(s)", request.url.path, len(details))
        return error_response(
            request,
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "validation_error",
            "Request validation failed",
            details,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        code = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed"}.get(
            exc.status_code, "http_error"
        )
        return error_response(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, _exc: Exception):  # pragma: no cover - safety net
        logger.exception("Unexpected error on %s %s", request.method, request.url.path)
        return error_response(request, 500, "internal_error", "Internal server error")
