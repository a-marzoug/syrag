from collections.abc import Awaitable, Callable, Sequence
from enum import Enum
from typing import Any

from fastapi import FastAPI
from starlette.requests import Request

from fastrag.protocols import (
    LLM,
    Chunker,
    Embedder,
    GenerationPolicy,
    PromptAssembler,
    VectorStore,
)
from fastrag.schemas import ErrorResponse, IngestRequest, IngestResponse, QueryRequest, RAGResponse
from fastrag.services import PipelineService, RetrievalStrategy

type IngestHandler = Callable[[IngestRequest], IngestRequest | Awaitable[IngestRequest]]
type QueryHandler = Callable[[QueryRequest], QueryRequest | Awaitable[QueryRequest]]
type OpenAPIResponses = dict[int | str, dict[str, Any]]

REQUEST_ID_HEADER = {
    "description": "Framework-generated correlation ID for the request.",
    "schema": {"type": "string"},
}
RETRY_AFTER_HEADER = {
    "description": "Suggested wait time before retrying a throttled request.",
    "schema": {"type": "string"},
}


def _query_openapi_extra() -> dict[str, Any]:
    return {
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "grounded_query": {
                            "summary": "Tenant-scoped grounded query",
                            "value": {
                                "query": "What does FastRAG provide for RAG services?",
                                "collection": "overview",
                                "tenant_id": "tenant-a",
                                "top_k": 3,
                                "filters": {"topic": "product"},
                            },
                        }
                    }
                }
            }
        }
    }


def _ingest_openapi_extra() -> dict[str, Any]:
    return {
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "document_batch": {
                            "summary": "Collection ingest request",
                            "value": {
                                "documents": [
                                    (
                                        "FastRAG is a production-first Python framework "
                                        "for RAG services."
                                    ),
                                    "It exposes typed query and ingest routes on top of FastAPI.",
                                ],
                                "collection": "overview",
                                "tenant_id": "tenant-a",
                                "metadata": {"source_id": "overview", "page_number": 1},
                            },
                        }
                    }
                }
            }
        }
    }


def _query_responses() -> OpenAPIResponses:
    return {
        200: {
            "description": "Grounded answer with citations and usage metadata.",
            "headers": {"x-request-id": REQUEST_ID_HEADER},
            "content": {
                "application/json": {
                    "examples": {
                        "grounded_answer": {
                            "summary": "Answer with citations",
                            "value": {
                                "answer": (
                                    "FastRAG provides a production-first Python framework "
                                    "for building RAG services."
                                ),
                                "citations": [
                                    {
                                        "source_id": "overview",
                                        "score": 0.98,
                                        "snippet": (
                                            "FastRAG is a production-first Python framework "
                                            "for RAG services."
                                        ),
                                        "page_number": 1,
                                    }
                                ],
                                "usage": {"prompt_tokens": 42, "completion_tokens": 18},
                            },
                        }
                    }
                }
            },
        },
        400: _error_response(
            description="Request validation or safety guard failure.",
            example_name="invalid_query",
            example_value={
                "error": {
                    "code": "query_too_large",
                    "message": "Query exceeds the configured safety limit.",
                    "stage": "safety",
                    "details": {
                        "max_query_characters": 4000,
                        "actual_query_characters": 5001,
                    },
                }
            },
        ),
        401: _error_response(
            description="Authentication hook rejected the request.",
            example_name="authentication_failed",
            example_value={
                "error": {
                    "code": "authentication_failed",
                    "message": "Failed to authenticate the request.",
                    "stage": "auth",
                    "details": {"component": "AuthHook"},
                }
            },
        ),
        429: _error_response(
            description="Rate limiter rejected the request.",
            example_name="rate_limited",
            example_value={
                "error": {
                    "code": "rate_limited",
                    "message": "Request rate limit exceeded.",
                    "stage": "rate_limit",
                    "details": {
                        "max_requests": 60,
                        "window_seconds": 60.0,
                        "retry_after_seconds": 12,
                    },
                }
            },
            extra_headers={"retry-after": RETRY_AFTER_HEADER},
        ),
        500: _error_response(
            description="Pipeline or provider failure during query execution.",
            example_name="generation_failed",
            example_value={
                "error": {
                    "code": "generation_failed",
                    "message": "Failed to generate a grounded response.",
                    "stage": "generate",
                    "details": {"component": "OpenAILLM"},
                }
            },
        ),
    }


def _ingest_responses() -> OpenAPIResponses:
    return {
        200: {
            "description": "Completed ingest result with collection and tenant scope.",
            "headers": {"x-request-id": REQUEST_ID_HEADER},
            "content": {
                "application/json": {
                    "examples": {
                        "ingest_completed": {
                            "summary": "Successful ingest",
                            "value": {
                                "status": "completed",
                                "ingested_documents": 2,
                                "collection": "overview",
                                "tenant_id": "tenant-a",
                            },
                        }
                    }
                }
            },
        },
        400: _error_response(
            description="Request validation or ingest safety guard failure.",
            example_name="too_many_documents",
            example_value={
                "error": {
                    "code": "too_many_documents",
                    "message": "Ingest request exceeds the configured document safety limit.",
                    "stage": "safety",
                    "details": {
                        "max_ingest_documents": 100,
                        "actual_ingest_documents": 125,
                    },
                }
            },
        ),
        401: _error_response(
            description="Authentication hook rejected the request.",
            example_name="authentication_failed",
            example_value={
                "error": {
                    "code": "authentication_failed",
                    "message": "Failed to authenticate the request.",
                    "stage": "auth",
                    "details": {"component": "AuthHook"},
                }
            },
        ),
        429: _error_response(
            description="Rate limiter rejected the request.",
            example_name="rate_limited",
            example_value={
                "error": {
                    "code": "rate_limited",
                    "message": "Request rate limit exceeded.",
                    "stage": "rate_limit",
                    "details": {
                        "max_requests": 60,
                        "window_seconds": 60.0,
                        "retry_after_seconds": 12,
                    },
                }
            },
            extra_headers={"retry-after": RETRY_AFTER_HEADER},
        ),
        500: _error_response(
            description="Pipeline or provider failure during ingest execution.",
            example_name="storage_failed",
            example_value={
                "error": {
                    "code": "storage_failed",
                    "message": "Failed to persist embedded documents.",
                    "stage": "store",
                    "details": {"component": "SQLiteVectorStore"},
                }
            },
        ),
    }


def _error_response(
    *,
    description: str,
    example_name: str,
    example_value: dict[str, Any],
    extra_headers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {"x-request-id": REQUEST_ID_HEADER}
    if extra_headers is not None:
        headers |= extra_headers
    return {
        "model": ErrorResponse,
        "description": description,
        "headers": headers,
        "content": {
            "application/json": {
                "examples": {
                    example_name: {
                        "value": example_value,
                    }
                }
            }
        },
    }


def _operation_description(*, default: str, handler: Callable[..., Any]) -> str:
    handler_description = (handler.__doc__ or "").strip()
    if not handler_description:
        return default
    return f"{default}\n\n{handler_description}"


def build_query_decorator(
    *,
    api: FastAPI,
    pipeline: PipelineService,
    path: str,
    embedder: Embedder,
    vector_store: VectorStore,
    llm: LLM,
    retrieval_strategy: RetrievalStrategy,
    prompt_assembler: PromptAssembler,
    generation_policy: GenerationPolicy,
    prepare_request: Callable[[Request, QueryRequest], Awaitable[QueryRequest]],
    resolve_request: Callable[[QueryRequest | Awaitable[QueryRequest]], Awaitable[QueryRequest]],
    tags: Sequence[str | Enum] | None = None,
) -> Callable[[QueryHandler], QueryHandler]:
    route_tags: list[str | Enum] = list(tags) if tags is not None else ["query"]

    def decorator(handler: QueryHandler) -> QueryHandler:
        async def endpoint(payload: QueryRequest, http_request: Request) -> RAGResponse:
            request = await prepare_request(http_request, payload)
            resolved_request = await resolve_request(handler(request))
            resolved_request = await prepare_request(http_request, resolved_request)
            return await pipeline.run_query(
                request=resolved_request,
                embedder=embedder,
                vector_store=vector_store,
                llm=llm,
                retrieval_strategy=retrieval_strategy,
                prompt_assembler=prompt_assembler,
                generation_policy=generation_policy,
            )

        endpoint.__name__ = handler.__name__
        endpoint.__doc__ = handler.__doc__
        api.post(
            path,
            response_model=RAGResponse,
            tags=route_tags,
            summary="Run a grounded query",
            description=_operation_description(
                default=(
                    "Resolve a grounded answer from the configured retrieval and "
                    "generation pipeline."
                ),
                handler=handler,
            ),
            response_description="Grounded answer with citations.",
            responses=_query_responses(),
            openapi_extra=_query_openapi_extra(),
        )(endpoint)
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
    prepare_request: Callable[[Request, IngestRequest], Awaitable[IngestRequest]],
    resolve_request: Callable[[IngestRequest | Awaitable[IngestRequest]], Awaitable[IngestRequest]],
    tags: Sequence[str | Enum] | None = None,
) -> Callable[[IngestHandler], IngestHandler]:
    route_tags: list[str | Enum] = list(tags) if tags is not None else ["ingest"]

    def decorator(handler: IngestHandler) -> IngestHandler:
        async def endpoint(payload: IngestRequest, http_request: Request) -> IngestResponse:
            request = await prepare_request(http_request, payload)
            resolved_request = await resolve_request(handler(request))
            resolved_request = await prepare_request(http_request, resolved_request)
            return await pipeline.run_ingest(
                request=resolved_request,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
            )

        endpoint.__name__ = handler.__name__
        endpoint.__doc__ = handler.__doc__
        api.post(
            path,
            response_model=IngestResponse,
            tags=route_tags,
            summary="Ingest source documents",
            description=_operation_description(
                default=(
                    "Normalize, chunk, embed, and store source documents in the configured "
                    "vector store."
                ),
                handler=handler,
            ),
            response_description="Completed ingest result.",
            responses=_ingest_responses(),
            openapi_extra=_ingest_openapi_extra(),
        )(endpoint)
        return handler

    return decorator
