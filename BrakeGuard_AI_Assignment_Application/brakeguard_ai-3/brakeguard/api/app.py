"""FastAPI application factory.

``create_app`` builds a fresh application with its own
:class:`ServiceContainer`; tests create isolated apps (with a tiny model,
without a model, with an API key...) without touching global state.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import __version__
from ..config import Settings, get_settings
from ..logging_config import configure_logging, get_logger
from ..models import ModelBundle
from ..pipeline import load_or_train
from .container import ServiceContainer
from .errors import register_exception_handlers
from .routes import api_router, ops_router

logger = get_logger(__name__)
HERE = Path(__file__).resolve().parent
_UNSET = object()


def create_app(settings: Settings | None = None, bundle: ModelBundle | None | object = _UNSET) -> FastAPI:
    """Build the API. ``bundle=None`` starts the service without a model."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not hasattr(app.state, "container"):
            model = load_or_train(settings) if bundle is _UNSET else bundle
            app.state.container = ServiceContainer(settings=settings, bundle=model)
        logger.info("%s API ready (model=%s)", settings.app_name, app.state.container.predictor.version)
        yield
        logger.info("%s API shutting down", settings.app_name)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Predicts the probability of a brake-system fault within the next "
            "30 days / 1,000 km from vehicle telemetry."
        ),
        lifespan=lifespan,
    )
    if bundle is not _UNSET:
        app.state.container = ServiceContainer(settings=settings, bundle=bundle)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID", str(uuid4()))
        start = perf_counter()
        response = await call_next(request)
        elapsed = (perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Process-Time-ms"] = f"{elapsed:.2f}"
        logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, response.status_code, elapsed)
        return response

    register_exception_handlers(app)
    app.include_router(ops_router)
    app.include_router(api_router)

    templates = Jinja2Templates(directory=str(HERE / "templates"))
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard(request: Request):
        c: ServiceContainer = request.app.state.container
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "vehicles": c.feature_store.vehicle_ids(),
                "metrics": c.bundle.metrics if c.bundle else {},
                "model_version": c.predictor.version,
                "auth_enabled": bool(settings.api_key),
            },
        )

    return app
