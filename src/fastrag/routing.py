from collections.abc import Awaitable, Callable, Sequence
from enum import Enum

from fastapi import FastAPI

from fastrag.protocols import LLM, Chunker, Embedder, VectorStore
from fastrag.schemas import IngestRequest, IngestResponse, QueryRequest, RAGResponse
from fastrag.services import PipelineService

type IngestHandler = Callable[[IngestRequest], IngestRequest | Awaitable[IngestRequest]]
type QueryHandler = Callable[[QueryRequest], QueryRequest | Awaitable[QueryRequest]]


def build_query_decorator(
    *,
    api: FastAPI,
    pipeline: PipelineService,
    path: str,
    embedder: Embedder,
    vector_store: VectorStore,
    llm: LLM,
    resolve_request: Callable[[QueryRequest | Awaitable[QueryRequest]], Awaitable[QueryRequest]],
    tags: Sequence[str | Enum] | None = None,
) -> Callable[[QueryHandler], QueryHandler]:
    route_tags: list[str | Enum] = list(tags) if tags is not None else ["query"]

    def decorator(handler: QueryHandler) -> QueryHandler:
        async def endpoint(request: QueryRequest) -> RAGResponse:
            resolved_request = await resolve_request(handler(request))
            return await pipeline.run_query(
                request=resolved_request,
                embedder=embedder,
                vector_store=vector_store,
                llm=llm,
            )

        endpoint.__name__ = handler.__name__
        endpoint.__doc__ = handler.__doc__
        api.post(path, response_model=RAGResponse, tags=route_tags)(endpoint)
        return handler

    return decorator


def build_ingest_decorator(
    *,
    api: FastAPI,
    pipeline: PipelineService,
    path: str,
    chunker: Chunker,
    embedder: Embedder,
    vector_store: VectorStore,
    resolve_request: Callable[[IngestRequest | Awaitable[IngestRequest]], Awaitable[IngestRequest]],
    tags: Sequence[str | Enum] | None = None,
) -> Callable[[IngestHandler], IngestHandler]:
    route_tags: list[str | Enum] = list(tags) if tags is not None else ["ingest"]

    def decorator(handler: IngestHandler) -> IngestHandler:
        async def endpoint(request: IngestRequest) -> IngestResponse:
            resolved_request = await resolve_request(handler(request))
            return await pipeline.run_ingest(
                request=resolved_request,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
            )

        endpoint.__name__ = handler.__name__
        endpoint.__doc__ = handler.__doc__
        api.post(path, response_model=IngestResponse, tags=route_tags)(endpoint)
        return handler

    return decorator
