"""FastAPI entrypoint for Flowcept webservice."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from flowcept.configs import WEBSERVER_HOST, WEBSERVER_PORT
from flowcept.webservice.routers.datasets import router as datasets_router
from flowcept.webservice.routers.health import router as health_router
from flowcept.webservice.routers.models import router as models_router
from flowcept.webservice.routers.objects import router as objects_router
from flowcept.webservice.routers.query import router as query_router
from flowcept.webservice.routers.tasks import router as tasks_router
from flowcept.webservice.routers.workflows import router as workflows_router


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Flowcept Webservice API",
        version="1.0.0",
        description=(
            "Read-only REST API for Flowcept provenance data. "
            "Provides workflows, tasks, and objects endpoints with query support."
        ),
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/", tags=["health"])
    def root() -> dict:
        return {
            "status": "up",
            "service": "flowcept-webservice",
            "host": WEBSERVER_HOST,
            "port": WEBSERVER_PORT,
        }

    app.include_router(health_router, prefix="/api/v1")
    app.include_router(workflows_router, prefix="/api/v1")
    app.include_router(tasks_router, prefix="/api/v1")
    app.include_router(objects_router, prefix="/api/v1")
    app.include_router(datasets_router, prefix="/api/v1")
    app.include_router(models_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")

    return app


app = create_app()
