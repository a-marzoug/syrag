from collections.abc import Awaitable, Callable, Sequence
from enum import Enum
from inspect import isawaitable
from typing import Any, cast

from fastapi import FastAPI
from fastapi.routing import APIRouter
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import Receive, Scope, Send

from fastrag.bootstrap import BootstrapService
from fastrag.config import ComponentDefaults, Settings, get_settings
from fastrag.errors import FastRAGError
from fastrag.observability import EventListener, ObservabilityHub
from fastrag.protocols import LLM, Embedder, VectorStore
from fastrag.registry import ComponentRegistry
from fastrag.schemas import (
    ErrorDetail,
    ErrorResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    RAGResponse,
)
from fastrag.services import PipelineService

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
        self.defaults = settings.defaults.model_copy(deep=True)
        self.registry = ComponentRegistry()
        self.bootstrap = BootstrapService(settings.bootstrap)
        self.observability = ObservabilityHub()
        self.pipeline = PipelineService(observability=self.observability)
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
        self.api.state.bootstrap = self.bootstrap
        self.api.state.defaults = self.defaults
        self.api.state.observability = self.observability
        self.api.state.pipeline = self.pipeline
        self.api.state.registry = self.registry
        self.bootstrap.apply(registry=self.registry, defaults=self.defaults)
        self._register_exception_handlers()
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
        embedder: Embedder | str | None = None,
        vector_store: VectorStore | str | None = None,
        llm: LLM | str | None = None,
        tags: Sequence[str | Enum] | None = None,
    ) -> Callable[[QueryHandler], QueryHandler]:
        resolved_embedder = self._resolve_embedder(embedder)
        resolved_vector_store = self._resolve_vector_store(vector_store)
        resolved_llm = self._resolve_llm(llm)
        route_tags: list[str | Enum] = list(tags) if tags is not None else ["query"]

        def decorator(handler: QueryHandler) -> QueryHandler:
            async def endpoint(request: QueryRequest) -> RAGResponse:
                resolved_request = await self._resolve_query_request(handler(request))
                return await self.pipeline.run_query(
                    request=resolved_request,
                    embedder=resolved_embedder,
                    vector_store=resolved_vector_store,
                    llm=resolved_llm,
                )

            endpoint.__name__ = handler.__name__
            endpoint.__doc__ = handler.__doc__
            self.api.post(path, response_model=RAGResponse, tags=route_tags)(endpoint)
            return handler

        return decorator

    def ingest(
        self,
        path: str,
        *,
        embedder: Embedder | str | None = None,
        vector_store: VectorStore | str | None = None,
        tags: Sequence[str | Enum] | None = None,
    ) -> Callable[[IngestHandler], IngestHandler]:
        resolved_embedder = self._resolve_embedder(embedder)
        resolved_vector_store = self._resolve_vector_store(vector_store)
        route_tags: list[str | Enum] = list(tags) if tags is not None else ["ingest"]

        def decorator(handler: IngestHandler) -> IngestHandler:
            async def endpoint(request: IngestRequest) -> IngestResponse:
                resolved_request = await self._resolve_ingest_request(handler(request))
                return await self.pipeline.run_ingest(
                    request=resolved_request,
                    embedder=resolved_embedder,
                    vector_store=resolved_vector_store,
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

    def configure_defaults(
        self,
        *,
        embedder: str | None = None,
        vector_store: str | None = None,
        llm: str | None = None,
    ) -> None:
        self.defaults = ComponentDefaults(
            embedder=embedder if embedder is not None else self.defaults.embedder,
            vector_store=(
                vector_store if vector_store is not None else self.defaults.vector_store
            ),
            llm=llm if llm is not None else self.defaults.llm,
        )
        self.api.state.defaults = self.defaults

    def add_event_listener(self, listener: EventListener) -> None:
        self.observability.add_listener(listener)

    def _register_exception_handlers(self) -> None:
        @self.api.exception_handler(FastRAGError)
        async def handle_fastrag_error(
            _request: Request,
            exc: FastRAGError,
        ) -> JSONResponse:
            response = ErrorResponse(
                error=ErrorDetail(
                    code=exc.code,
                    message=exc.message,
                    stage=exc.stage,
                    details=exc.details,
                )
            )
            return JSONResponse(status_code=exc.status_code, content=response.model_dump())

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

    def _resolve_embedder(self, component: Embedder | str | None) -> Embedder:
        if component is None:
            default_name = self._require_default_name(
                component_name="embedder",
                configured_name=self.defaults.embedder,
            )
            return self.registry.get_embedder(default_name)
        if isinstance(component, str):
            return self.registry.get_embedder(component)
        self._validate_component(component, Embedder, "embedder")
        return component

    def _resolve_vector_store(self, component: VectorStore | str | None) -> VectorStore:
        if component is None:
            default_name = self._require_default_name(
                component_name="vector_store",
                configured_name=self.defaults.vector_store,
            )
            return self.registry.get_vector_store(default_name)
        if isinstance(component, str):
            return self.registry.get_vector_store(component)
        self._validate_component(component, VectorStore, "vector_store")
        return component

    def _resolve_llm(self, component: LLM | str | None) -> LLM:
        if component is None:
            default_name = self._require_default_name(
                component_name="llm",
                configured_name=self.defaults.llm,
            )
            return self.registry.get_llm(default_name)
        if isinstance(component, str):
            return self.registry.get_llm(component)
        self._validate_component(component, LLM, "llm")
        return component

    def _require_default_name(
        self,
        *,
        component_name: str,
        configured_name: str | None,
    ) -> str:
        if configured_name is not None:
            return configured_name

        msg = f"No default {component_name} configured for this app"
        raise ValueError(msg)

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
