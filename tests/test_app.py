import pytest
from httpx import ASGITransport, AsyncClient

from syrag.app import SyRAG, create_app


def test_create_app_returns_syrag_wrapper() -> None:
    application = create_app()

    assert isinstance(application, SyRAG)
    assert application.api.title == "SyRAG"
    assert application.api.version == "0.1.0"


@pytest.mark.asyncio
async def test_healthcheck_returns_service_metadata() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=create_app().api),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "environment": "development",
        "service": "SyRAG",
    }
