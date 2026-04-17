import pytest
from httpx import ASGITransport, AsyncClient

from fastrag.app import create_app


@pytest.mark.asyncio
async def test_healthcheck_returns_service_metadata() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=create_app()),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "environment": "development",
        "service": "FastRAG",
    }
