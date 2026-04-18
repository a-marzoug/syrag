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
from fastrag.schemas import QueryRequest, RAGResponse

ExceptionHandler = Callable[[Request, Exception], Awaitable[Response]]
ExceptionHandlerDecorator = Callable[[ExceptionHandler], ExceptionHandler]
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
        embedder: Embedder,
        vector_store: VectorStore,
        llm: LLM,
        tags: Sequence[str | Enum] | None = None,
    ) -> Callable[[QueryHandler], QueryHandler]:
        self._validate_query_components(
            embedder=embedder,
            vector_store=vector_store,
            llm=llm,
        )
        route_tags: list[str | Enum] = list(tags) if tags is not None else ["query"]

        def decorator(handler: QueryHandler) -> QueryHandler:
            async def endpoint(request: QueryRequest) -> RAGResponse:
                resolved_request = await self._resolve_query_request(handler(request))
                query_embedding = (await embedder.embed([resolved_request.query]))[0]
                context = await vector_store.query(
                    query_embedding=query_embedding,
                    top_k=resolved_request.top_k,
                    collection=resolved_request.collection,
                    tenant_id=resolved_request.tenant_id,
                    filters=resolved_request.filters,
                )
                return await llm.generate(query=resolved_request, context=context)

            endpoint.__name__ = handler.__name__
            endpoint.__doc__ = handler.__doc__
            self.api.post(path, response_model=RAGResponse, tags=route_tags)(endpoint)
            return handler

        return decorator

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

    def _validate_query_components(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        llm: LLM,
    ) -> None:
        self._validate_component(embedder, Embedder, "embedder")
        self._validate_component(vector_store, VectorStore, "vector_store")
        self._validate_component(llm, LLM, "llm")

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
