import uvicorn

from fastrag.config import get_settings


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "fastrag.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
