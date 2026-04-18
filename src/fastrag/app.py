from collections.abc import Awaitable, Callable, Sequence
from enum import Enum
from inspect import isawaitable
from typing import Any, cast

from fastapi import FastAPI
from fastapi.routing import APIRouter
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from fastrag.config import Settings, get_settings
from fastrag.protocols import LLM, Embedder, VectorStore
from fastrag.registry import ComponentRegistry
from fastrag.schemas import IngestRequest, IngestResponse, QueryRequest, RAGResponse

ExceptionHandler = Callable[[Request, Exception], Awaitable[Response]]
ExceptionHandlerDecorator = Callable[[ExceptionHandler], ExceptionHandler]
IngestHandler = Callable[[IngestRequest], IngestRequest | Awaitable[IngestRequest]]
QueryHandler = Callable[[QueryRequest], QueryRequest | Awaitable[QueryRequest]]


class FastRAG:
    """Application wrapper exposing a stable framework entry point."""

    def __init__(
        self,
        *,
        title: str,
        version: str,
        description: str,
        settings: Settings,
        docs_url: str = "/docs",
        redoc_url: str = "/redoc",
        openapi_url: str = "/openapi.json",
        middleware: list[Middleware] | None = None,
    ) -> None:
        self.settings = settings
        self.registry = ComponentRegistry()
        self.api = FastAPI(
            title=title,
            version=version,
            description=description,
            docs_url=docs_url,
            redoc_url=redoc_url,
            openapi_url=openapi_url,
            middleware=middleware,
        )
        self.api.state.fastrag = self
        self.api.state.registry = self.registry
        self._register_system_routes()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.api(scope, receive, send)

    @property
    def router(self) -> APIRouter:
        return self.api.router

    def get(
        self,
        path: str,
        *,
        tags: Sequence[str | Enum] | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.api.get(path, tags=list(tags) if tags is not None else None, **kwargs)

    def post(
        self,
        path: str,
        *,
        tags: Sequence[str | Enum] | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.api.post(path, tags=list(tags) if tags is not None else None, **kwargs)

    def include_router(self, router: APIRouter, **kwargs: Any) -> None:
        self.api.include_router(router, **kwargs)

    def add_api_route(self, path: str, endpoint: Callable[..., Any], **kwargs: Any) -> None:
        self.api.add_api_route(path, endpoint, **kwargs)

    def add_middleware(self, middleware_class: type[Any], **kwargs: Any) -> None:
        self.api.add_middleware(cast(Any, middleware_class), **kwargs)

    def exception_handler(
        self,
        exc_class_or_status_code: int | type[Exception],
    ) -> ExceptionHandlerDecorator:
        return self.api.exception_handler(exc_class_or_status_code)

    def query(
        self,
        path: str,
        *,
        embedder: Embedder | str,
        vector_store: VectorStore | str,
        llm: LLM | str,
        tags: Sequence[str | Enum] | None = None,
    ) -> Callable[[QueryHandler], QueryHandler]:
        resolved_embedder = self._resolve_embedder(embedder)
        resolved_vector_store = self._resolve_vector_store(vector_store)
        resolved_llm = self._resolve_llm(llm)
        route_tags: list[str | Enum] = list(tags) if tags is not None else ["query"]

        def decorator(handler: QueryHandler) -> QueryHandler:
            async def endpoint(request: QueryRequest) -> RAGResponse:
                resolved_request = await self._resolve_query_request(handler(request))
                query_embedding = (await resolved_embedder.embed([resolved_request.query]))[0]
                context = await resolved_vector_store.query(
                    query_embedding=query_embedding,
                    top_k=resolved_request.top_k,
                    collection=resolved_request.collection,
                    tenant_id=resolved_request.tenant_id,
                    filters=resolved_request.filters,
                )
                return await resolved_llm.generate(query=resolved_request, context=context)

            endpoint.__name__ = handler.__name__
            endpoint.__doc__ = handler.__doc__
            self.api.post(path, response_model=RAGResponse, tags=route_tags)(endpoint)
            return handler

        return decorator

    def ingest(
        self,
        path: str,
        *,
        embedder: Embedder | str,
        vector_store: VectorStore | str,
        tags: Sequence[str | Enum] | None = None,
    ) -> Callable[[IngestHandler], IngestHandler]:
        resolved_embedder = self._resolve_embedder(embedder)
        resolved_vector_store = self._resolve_vector_store(vector_store)
        route_tags: list[str | Enum] = list(tags) if tags is not None else ["ingest"]

        def decorator(handler: IngestHandler) -> IngestHandler:
            async def endpoint(request: IngestRequest) -> IngestResponse:
                resolved_request = await self._resolve_ingest_request(handler(request))
                embeddings = await resolved_embedder.embed(resolved_request.documents)
                metadata = self._expand_ingest_metadata(resolved_request)
                await resolved_vector_store.upsert(
                    documents=resolved_request.documents,
                    embeddings=embeddings,
                    collection=resolved_request.collection,
                    tenant_id=resolved_request.tenant_id,
                    metadata=metadata,
                )
                return IngestResponse(
                    status="completed",
                    ingested_documents=len(resolved_request.documents),
                    collection=resolved_request.collection,
                    tenant_id=resolved_request.tenant_id,
                )

            endpoint.__name__ = handler.__name__
            endpoint.__doc__ = handler.__doc__
            self.api.post(path, response_model=IngestResponse, tags=route_tags)(endpoint)
            return handler

        return decorator

    def register_embedder(self, name: str, component: Embedder) -> None:
        self.registry.register_embedder(name, component)

    def register_vector_store(self, name: str, component: VectorStore) -> None:
        self.registry.register_vector_store(name, component)

    def register_llm(self, name: str, component: LLM) -> None:
        self.registry.register_llm(name, component)

    def _register_system_routes(self) -> None:
        @self.api.get("/health", tags=["system"])
        async def healthcheck() -> dict[str, str]:
            return {
                "status": "ok",
                "environment": self.settings.environment,
                "service": self.settings.app_name,
            }

    async def _resolve_query_request(
        self,
        result: QueryRequest | Awaitable[QueryRequest],
    ) -> QueryRequest:
        if isawaitable(result):
            resolved_result = await result
        else:
            resolved_result = result

        return resolved_result

    async def _resolve_ingest_request(
        self,
        result: IngestRequest | Awaitable[IngestRequest],
    ) -> IngestRequest:
        if isawaitable(result):
            resolved_result = await result
        else:
            resolved_result = result

        return resolved_result

    def _expand_ingest_metadata(self, request: IngestRequest) -> list[dict[str, Any]]:
        if len(request.documents) == 1:
            return [dict(request.metadata)]

        source_id = request.metadata.get("source_id")
        metadata_items: list[dict[str, Any]] = []
        for index, _document in enumerate(request.documents):
            metadata = dict(request.metadata)
            if isinstance(source_id, str) and source_id.strip():
                metadata["source_id"] = f"{source_id.strip()}-{index}"
            metadata_items.append(metadata)

        return metadata_items

    def _resolve_embedder(self, component: Embedder | str) -> Embedder:
        if isinstance(component, str):
            return self.registry.get_embedder(component)
        self._validate_component(component, Embedder, "embedder")
        return component

    def _resolve_vector_store(self, component: VectorStore | str) -> VectorStore:
        if isinstance(component, str):
            return self.registry.get_vector_store(component)
        self._validate_component(component, VectorStore, "vector_store")
        return component

    def _resolve_llm(self, component: LLM | str) -> LLM:
        if isinstance(component, str):
            return self.registry.get_llm(component)
        self._validate_component(component, LLM, "llm")
        return component

    def _validate_component(
        self,
        component: object,
        protocol: type[object],
        component_name: str,
    ) -> None:
        if isinstance(component, protocol):
            return

        msg = f"{component_name} must implement the {protocol.__name__} protocol"
        raise TypeError(msg)


def create_app(settings: Settings | None = None) -> FastRAG:
    resolved_settings = settings or get_settings()

    return FastRAG(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        description="Production-first FastRAG application bootstrap.",
        settings=resolved_settings,
    )


app = create_app()
