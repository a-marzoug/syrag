from fastrag._optional import missing_optional_dependency
from fastrag.app import create_app
from fastrag.config import get_settings


def run() -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via runtime path
        raise missing_optional_dependency(
            feature="fastrag server CLI",
            extra="server",
        ) from exc

    settings = get_settings()
    application = create_app(settings)
    uvicorn.run(
        application.api,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
