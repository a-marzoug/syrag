import uvicorn

from fastrag.app import create_app
from fastrag.config import get_settings


def run() -> None:
    settings = get_settings()
    application = create_app(settings)
    uvicorn.run(
        application.api,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
