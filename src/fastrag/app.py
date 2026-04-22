import logging
from collections.abc import Awaitable, Callable, Sequence
from enum import Enum
from inspect import isawaitable
from time import perf_counter
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
from fastrag.guardrails import DefaultSafetyGuard, InMemoryRateLimiter
from fastrag.hooks import DefaultRequestContextHook, NoOpAuthHook
from fastrag.observability import EventListener, ObservabilityHub
from fastrag.protocols import (
    LLM,
    AuthHook,
    Chunker,
    Embedder,
    GenerationPolicy,
    PromptAssembler,
    RateLimiter,
    RequestContextHook,
    SafetyGuard,
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
    RequestContext,
)
from fastrag.services import (
    DefaultGenerationPolicy,
    DefaultPromptAssembler,
    DefaultRetrievalStrategy,
    PipelineService,
    RetrievalStrategy,
)
from fastrag.structured_logging import StructuredLogging
from fastrag.tracing import OpenTelemetryTracer, OpenTelemetryTracing

type ExceptionHandler = Callable[[Request, Exception], Awaitable[Response]]
type ExceptionHandlerDecorator = Callable[[ExceptionHandler], ExceptionHandler]


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
        self.auth_hook: AuthHook = NoOpAuthHook()
        self.chunker = PassThroughChunker()
        self.generation_policy = DefaultGenerationPolicy()
        self.prompt_assembler = DefaultPromptAssembler()
        self.rate_limiter: RateLimiter | None = None
        self.request_context_hook: RequestContextHook = DefaultRequestContextHook()
        self.retrieval_strategy = DefaultRetrievalStrategy(observability=self.observability)
        self.safety_guard: SafetyGuard = DefaultSafetyGuard()
        self.structured_logging: StructuredLogging | None = None
        self.tracing: OpenTelemetryTracing | None = None
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
        self.api.state.auth_hook = self.auth_hook
        self.api.state.chunker = self.chunker
        self.api.state.generation_policy = self.generation_policy
        self.api.state.prompt_assembler = self.prompt_assembler
        self.api.state.rate_limiter = self.rate_limiter
        self.api.state.request_context_hook = self.request_context_hook
        self.api.state.retrieval_strategy = self.retrieval_strategy
        self.api.state.registry = self.registry
        self.api.state.resolver = self.resolver
        self.api.state.safety_guard = self.safety_guard
        self.api.state.structured_logging = self.structured_logging
        self.api.state.tracing = self.tracing
        self.bootstrap.apply(registry=self.registry, defaults=self.defaults)
        self._register_exception_handlers()
        self._register_request_context_middleware()
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
            prepare_request=self._prepare_query_request,
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
            prepare_request=self._prepare_ingest_request,
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

    def configure_logging(
        self,
        *,
        structured_logging: StructuredLogging | None = None,
        logger: logging.Logger | None = None,
    ) -> StructuredLogging:
        resolved_logging = structured_logging or StructuredLogging(logger=logger)
        if self.structured_logging is not None:
            self.observability.remove_listener(self.structured_logging.listener)
        self.structured_logging = resolved_logging
        self.observability.add_listener(resolved_logging.listener)
        self.api.state.structured_logging = resolved_logging
        return resolved_logging

    def configure_tracing(
        self,
        *,
        tracing: OpenTelemetryTracing | None = None,
        tracer: OpenTelemetryTracer | None = None,
        instrumentation_scope: str = "fastrag",
    ) -> OpenTelemetryTracing:
        resolved_tracing = tracing or OpenTelemetryTracing(
            tracer=tracer,
            instrumentation_scope=instrumentation_scope,
        )
        if self.tracing is not None:
            self.observability.remove_listener(self.tracing.listener)
        self.tracing = resolved_tracing
        self.observability.add_listener(resolved_tracing.listener)
        self.api.state.tracing = resolved_tracing
        return resolved_tracing

    def configure_rate_limiting(
        self,
        *,
        max_requests: int,
        window_seconds: float = 60.0,
        include_path: bool = True,
    ) -> InMemoryRateLimiter:
        rate_limiter = InMemoryRateLimiter(
            max_requests=max_requests,
            window_seconds=window_seconds,
            include_path=include_path,
        )
        self.set_rate_limiter(rate_limiter)
        return rate_limiter

    def set_request_context_hook(self, hook: RequestContextHook) -> None:
        if not isinstance(hook, RequestContextHook):
            msg = "request_context_hook must implement the RequestContextHook protocol"
            raise TypeError(msg)
        self.request_context_hook = hook
        self.api.state.request_context_hook = hook

    def set_auth_hook(self, hook: AuthHook) -> None:
        if not isinstance(hook, AuthHook):
            msg = "auth_hook must implement the AuthHook protocol"
            raise TypeError(msg)
        self.auth_hook = hook
        self.api.state.auth_hook = hook

    def set_rate_limiter(self, rate_limiter: RateLimiter | None) -> None:
        if rate_limiter is not None and not isinstance(rate_limiter, RateLimiter):
            msg = "rate_limiter must implement the RateLimiter protocol"
            raise TypeError(msg)
        self.rate_limiter = rate_limiter
        self.api.state.rate_limiter = rate_limiter

    def set_safety_guard(self, safety_guard: SafetyGuard) -> None:
        if not isinstance(safety_guard, SafetyGuard):
            msg = "safety_guard must implement the SafetyGuard protocol"
            raise TypeError(msg)
        self.safety_guard = safety_guard
        self.api.state.safety_guard = safety_guard

    def get_request_context(self, request: Request) -> RequestContext:
        context = getattr(request.state, "fastrag_context", None)
        if isinstance(context, RequestContext):
            return context

        msg = "Request context is not available for this request"
        raise RuntimeError(msg)

    def get_tenant_id(self, request: Request) -> str | None:
        return self.get_request_context(request).tenant_id

    def _register_exception_handlers(self) -> None:
        @self.api.exception_handler(FastRAGError)
        async def handle_fastrag_error(
            _request: Request,
            exc: FastRAGError,
        ) -> JSONResponse:
            return self._build_error_response(exc)

    def _build_error_response(self, exc: FastRAGError) -> JSONResponse:
        response = ErrorResponse(
            error=ErrorDetail(
                code=exc.code,
                message=exc.message,
                stage=exc.stage,
                details=exc.details,
            )
        )
        http_response = JSONResponse(status_code=exc.status_code, content=response.model_dump())
        retry_after_seconds = exc.details.get("retry_after_seconds")
        if retry_after_seconds is not None:
            http_response.headers.setdefault("retry-after", str(retry_after_seconds))
        return http_response

    def _register_request_context_middleware(self) -> None:
        @self.api.middleware("http")
        async def request_context_middleware(
            request: Request,
            call_next: Callable[[Request], Awaitable[Response]],
        ) -> Response:
            context = RequestContext()
            request_started_at = perf_counter()
            structured_logging = self.structured_logging
            tracing = self.tracing
            if tracing is None:
                response_for_logging: Response | None = None
                try:
                    try:
                        context = await self._build_request_context(
                            request=request,
                            context=context,
                        )
                        await self._enforce_rate_limit(request=request, context=context)
                    except FastRAGError as exc:
                        response_for_logging = self._build_error_response(exc)
                        self._apply_request_headers(
                            response=response_for_logging,
                            context=context,
                        )
                        return response_for_logging
                    response_for_logging = await call_next(request)
                    self._apply_request_headers(response=response_for_logging, context=context)
                    return response_for_logging
                finally:
                    if structured_logging is not None:
                        structured_logging.log_request(
                            request=request,
                            context=context,
                            response=response_for_logging,
                            duration_ms=(perf_counter() - request_started_at) * 1000,
                        )

            with tracing.start_request_span(request=request) as request_span:
                response: Response | None = None
                try:
                    context = await self._build_request_context(request=request, context=context)
                    await self._enforce_rate_limit(request=request, context=context)
                    tracing.enrich_request_span(span=request_span, context=context)
                    response = await call_next(request)
                    self._apply_request_headers(response=response, context=context)
                    return response
                except FastRAGError as exc:
                    tracing.enrich_request_span(span=request_span, context=context)
                    tracing.record_exception(span=request_span, exception=exc)
                    response = self._build_error_response(exc)
                    self._apply_request_headers(response=response, context=context)
                    return response
                except Exception as exc:
                    tracing.enrich_request_span(span=request_span, context=context)
                    tracing.record_exception(span=request_span, exception=exc)
                    raise
                finally:
                    tracing.finish_request_span(span=request_span, response=response)
                    if structured_logging is not None:
                        structured_logging.log_request(
                            request=request,
                            context=context,
                            response=response,
                            duration_ms=(perf_counter() - request_started_at) * 1000,
                        )

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

    async def _prepare_query_request(
        self,
        request: Request,
        payload: QueryRequest,
    ) -> QueryRequest:
        bound_request = self._bind_request_tenant_context(request=request, payload=payload)
        context = self.get_request_context(request)
        return await self.safety_guard.validate_query(
            request=request,
            payload=bound_request,
            context=context,
        )

    async def _prepare_ingest_request(
        self,
        request: Request,
        payload: IngestRequest,
    ) -> IngestRequest:
        bound_request = self._bind_request_tenant_context(request=request, payload=payload)
        context = self.get_request_context(request)
        return await self.safety_guard.validate_ingest(
            request=request,
            payload=bound_request,
            context=context,
        )

    def _bind_request_tenant_context[RequestModel: (QueryRequest, IngestRequest)](
        self,
        *,
        request: Request,
        payload: RequestModel,
    ) -> RequestModel:
        context = self.get_request_context(request)
        tenant_id = self._resolve_request_tenant(
            context_tenant_id=context.tenant_id,
            payload_tenant_id=payload.tenant_id,
        )
        if tenant_id != context.tenant_id:
            request.state.fastrag_context = context.model_copy(update={"tenant_id": tenant_id})
        if tenant_id == payload.tenant_id:
            return payload
        return payload.model_copy(update={"tenant_id": tenant_id})

    def _resolve_request_tenant(
        self,
        *,
        context_tenant_id: str | None,
        payload_tenant_id: str | None,
    ) -> str | None:
        if context_tenant_id is None:
            return payload_tenant_id
        if payload_tenant_id is None or payload_tenant_id == context_tenant_id:
            return context_tenant_id

        raise FastRAGError(
            code="tenant_mismatch",
            message="Request tenant does not match the scoped tenant context.",
            stage="request",
            status_code=400,
            details={
                "context_tenant_id": context_tenant_id,
                "request_tenant_id": payload_tenant_id,
            },
        )

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

    async def _build_request_context(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> RequestContext:
        resolved_context = await self.request_context_hook.enrich(
            request=request,
            context=context,
        )
        resolved_context = await self.auth_hook.authenticate(
            request=request,
            context=resolved_context,
        )
        if resolved_context.request_id is None:
            resolved_context = resolved_context.model_copy(update={"request_id": "unknown"})
        request.state.fastrag_context = resolved_context
        return resolved_context

    async def _enforce_rate_limit(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> None:
        if self.rate_limiter is None:
            return
        await self.rate_limiter.check(request=request, context=context)

    def _apply_request_headers(
        self,
        *,
        response: Response,
        context: RequestContext,
    ) -> None:
        response.headers.setdefault("x-request-id", context.request_id or "unknown")


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
