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
from fastrag.dependencies import ComponentResolver
from fastrag.errors import FastRAGError
from fastrag.observability import EventListener, ObservabilityHub
from fastrag.protocols import (
    LLM,
    Chunker,
    Embedder,
    GenerationPolicy,
    PromptAssembler,
    VectorStore,
)
from fastrag.providers import PassThroughChunker, ProviderFactory
from fastrag.registry import ComponentRegistry
from fastrag.routing import (
    IngestHandler,
    QueryHandler,
    build_ingest_decorator,
    build_query_decorator,
)
from fastrag.schemas import (
    ErrorDetail,
    ErrorResponse,
    IngestRequest,
    QueryRequest,
)
from fastrag.services import (
    DefaultGenerationPolicy,
    DefaultPromptAssembler,
    DefaultRetrievalStrategy,
    PipelineService,
    RetrievalStrategy,
)

ExceptionHandler = Callable[[Request, Exception], Awaitable[Response]]
ExceptionHandlerDecorator = Callable[[ExceptionHandler], ExceptionHandler]


class FastRAG:
    """Application wrapper exposing a stable framework entry point."""

    def __init__(
        self,
        *,
        title: str,
        version: str,
        description: str,
        settings: Settings,
        provider_factory: ProviderFactory | None = None,
        docs_url: str = "/docs",
        redoc_url: str = "/redoc",
        openapi_url: str = "/openapi.json",
        middleware: list[Middleware] | None = None,
    ) -> None:
        self.settings = settings
        self.defaults = settings.defaults.model_copy(deep=True)
        self.registry = ComponentRegistry()
        self.resolver = ComponentResolver(registry=self.registry, defaults=self.defaults)
        self.bootstrap = BootstrapService(
            settings.bootstrap,
            provider_settings=settings.providers,
            factory=provider_factory,
        )
        self.observability = ObservabilityHub()
        self.pipeline = PipelineService(observability=self.observability)
        self.chunker = PassThroughChunker()
        self.generation_policy = DefaultGenerationPolicy()
        self.prompt_assembler = DefaultPromptAssembler()
        self.retrieval_strategy = DefaultRetrievalStrategy(observability=self.observability)
        self.pipeline.generation_policy = self.generation_policy
        self.pipeline.prompt_assembler = self.prompt_assembler
        self.pipeline.retrieval_strategy = self.retrieval_strategy
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
        self.api.state.provider_factory = self.bootstrap.factory
        self.api.state.provider_settings = self.bootstrap.provider_settings
        self.api.state.chunker = self.chunker
        self.api.state.generation_policy = self.generation_policy
        self.api.state.prompt_assembler = self.prompt_assembler
        self.api.state.retrieval_strategy = self.retrieval_strategy
        self.api.state.registry = self.registry
        self.api.state.resolver = self.resolver
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
        retrieval_strategy: RetrievalStrategy | None = None,
        prompt_assembler: PromptAssembler | None = None,
        generation_policy: GenerationPolicy | None = None,
        tags: Sequence[str | Enum] | None = None,
    ) -> Callable[[QueryHandler], QueryHandler]:
        resolved_embedder = self.resolver.resolve_embedder(embedder)
        resolved_vector_store = self.resolver.resolve_vector_store(vector_store)
        resolved_llm = self.resolver.resolve_llm(llm)
        resolved_retrieval_strategy = self._resolve_retrieval_strategy(retrieval_strategy)
        resolved_prompt_assembler = self._resolve_prompt_assembler(prompt_assembler)
        resolved_generation_policy = self._resolve_generation_policy(generation_policy)
        return build_query_decorator(
            api=self.api,
            pipeline=self.pipeline,
            path=path,
            embedder=resolved_embedder,
            vector_store=resolved_vector_store,
            llm=resolved_llm,
            retrieval_strategy=resolved_retrieval_strategy,
            prompt_assembler=resolved_prompt_assembler,
            generation_policy=resolved_generation_policy,
            resolve_request=self._resolve_query_request,
            tags=tags,
        )

    def ingest(
        self,
        path: str,
        *,
        chunker: Chunker | None = None,
        embedder: Embedder | str | None = None,
        vector_store: VectorStore | str | None = None,
        tags: Sequence[str | Enum] | None = None,
    ) -> Callable[[IngestHandler], IngestHandler]:
        resolved_chunker = self._resolve_chunker(chunker)
        resolved_embedder = self.resolver.resolve_embedder(embedder)
        resolved_vector_store = self.resolver.resolve_vector_store(vector_store)
        return build_ingest_decorator(
            api=self.api,
            pipeline=self.pipeline,
            path=path,
            chunker=resolved_chunker,
            embedder=resolved_embedder,
            vector_store=resolved_vector_store,
            resolve_request=self._resolve_ingest_request,
            tags=tags,
        )

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
        self.resolver.update_defaults(self.defaults)
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

    def _resolve_chunker(self, chunker: Chunker | None) -> Chunker:
        if chunker is None:
            return self.chunker
        if isinstance(chunker, Chunker):
            return chunker

        msg = "chunker must implement the Chunker protocol"
        raise TypeError(msg)

    def _resolve_retrieval_strategy(
        self,
        retrieval_strategy: RetrievalStrategy | None,
    ) -> RetrievalStrategy:
        if retrieval_strategy is None:
            return self.retrieval_strategy
        if isinstance(retrieval_strategy, RetrievalStrategy):
            return retrieval_strategy

        msg = "retrieval_strategy must implement the RetrievalStrategy protocol"
        raise TypeError(msg)

    def _resolve_prompt_assembler(
        self,
        prompt_assembler: PromptAssembler | None,
    ) -> PromptAssembler:
        if prompt_assembler is None:
            return self.prompt_assembler
        if isinstance(prompt_assembler, PromptAssembler):
            return prompt_assembler

        msg = "prompt_assembler must implement the PromptAssembler protocol"
        raise TypeError(msg)

    def _resolve_generation_policy(
        self,
        generation_policy: GenerationPolicy | None,
    ) -> GenerationPolicy:
        if generation_policy is None:
            return self.generation_policy
        if isinstance(generation_policy, GenerationPolicy):
            return generation_policy

        msg = "generation_policy must implement the GenerationPolicy protocol"
        raise TypeError(msg)

def create_app(
    settings: Settings | None = None,
    *,
    provider_factory: ProviderFactory | None = None,
) -> FastRAG:
    resolved_settings = settings or get_settings()

    return FastRAG(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        description="Production-first FastRAG application bootstrap.",
        settings=resolved_settings,
        provider_factory=provider_factory,
    )


app = create_app()
