from __future__ import annotations

from uuid import uuid4

from starlette.requests import Request

from fastrag.schemas import RequestContext


class DefaultRequestContextHook:
    """Populates base request metadata such as request IDs and HTTP details."""

    def __init__(
        self,
        request_id_header: str = "x-request-id",
        tenant_id_header: str = "x-tenant-id",
    ) -> None:
        self.request_id_header = request_id_header
        self.tenant_id_header = tenant_id_header

    async def enrich(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> RequestContext:
        request_id = context.request_id or request.headers.get(self.request_id_header) or str(
            uuid4()
        )
        tenant_id = context.tenant_id or request.headers.get(self.tenant_id_header)
        metadata = {
            **context.metadata,
            "method": request.method,
            "path": request.url.path,
        }
        return context.model_copy(
            update={
                "request_id": request_id,
                "tenant_id": tenant_id,
                "metadata": metadata,
            }
        )


class NoOpAuthHook:
    """Default authentication hook that leaves request context unchanged."""

    async def authenticate(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> RequestContext:
        return context
