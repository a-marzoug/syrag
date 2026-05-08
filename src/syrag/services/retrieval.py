from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from syrag.errors import PipelineStageError
from syrag.observability import ObservabilityHub, PipelineEvent
from syrag.protocols import EmbeddingVector, Reranker, VectorStore
from syrag.schemas import QueryRequest, RetrievedChunk


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Contract for framework-managed retrieval execution."""

    async def retrieve(
        self,
        *,
        request: QueryRequest,
        query_embedding: EmbeddingVector,
        vector_store: VectorStore,
    ) -> list[RetrievedChunk]:
        """Resolve retrieval context for a query embedding."""


class DefaultRetrievalStrategy:
    """Default retrieval strategy backed by a vector store query."""

    def __init__(self, observability: ObservabilityHub) -> None:
        self.observability = observability

    async def retrieve(
        self,
        *,
        request: QueryRequest,
        query_embedding: EmbeddingVector,
        vector_store: VectorStore,
    ) -> list[RetrievedChunk]:
        self._emit(
            status="started",
            component=type(vector_store).__name__,
        )
        try:
            context = await vector_store.query(
                query_embedding=query_embedding,
                top_k=request.top_k,
                collection=request.collection,
                tenant_id=request.tenant_id,
                filters=request.filters,
            )
            self._emit(
                status="succeeded",
                component=type(vector_store).__name__,
                details={"results": len(context)},
            )
            return context
        except Exception as exc:
            self._emit_failure(
                component=type(vector_store).__name__,
                error=exc,
            )
            raise PipelineStageError(
                code="retrieval_failed",
                message="Failed to retrieve supporting documents.",
                stage="retrieve",
                details={"component": type(vector_store).__name__},
            ) from exc

    def _emit(
        self,
        *,
        status: str,
        component: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.observability.emit(
            PipelineEvent(
                operation="query",
                stage="retrieve",
                status=status,
                component=component,
                details=details or {},
            )
        )

    def _emit_failure(
        self,
        *,
        component: str,
        error: Exception,
    ) -> None:
        self._emit(
            status="failed",
            component=component,
            details={"error_type": type(error).__name__},
        )


class RerankingRetrievalStrategy:
    """Retrieval strategy that reranks context returned by another strategy."""

    def __init__(
        self,
        *,
        base_strategy: RetrievalStrategy,
        reranker: Reranker,
        observability: ObservabilityHub | None = None,
    ) -> None:
        self.base_strategy = base_strategy
        self.reranker = reranker
        self.observability = observability or ObservabilityHub()

    async def retrieve(
        self,
        *,
        request: QueryRequest,
        query_embedding: EmbeddingVector,
        vector_store: VectorStore,
    ) -> list[RetrievedChunk]:
        context = await self.base_strategy.retrieve(
            request=request,
            query_embedding=query_embedding,
            vector_store=vector_store,
        )
        if not context:
            return []

        self._emit(
            status="started",
            component=type(self.reranker).__name__,
            details={"candidates": len(context)},
        )
        try:
            reranked_context = await self.reranker.rerank(
                query=request,
                context=context,
            )
            reranked_context = list(reranked_context[: request.top_k])
            self._emit(
                status="succeeded",
                component=type(self.reranker).__name__,
                details={"results": len(reranked_context)},
            )
            return reranked_context
        except Exception as exc:
            self._emit_failure(
                component=type(self.reranker).__name__,
                error=exc,
            )
            raise PipelineStageError(
                code="reranking_failed",
                message="Failed to rerank retrieved context.",
                stage="rerank",
                details={"component": type(self.reranker).__name__},
            ) from exc

    def _emit(
        self,
        *,
        status: str,
        component: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.observability.emit(
            PipelineEvent(
                operation="query",
                stage="rerank",
                status=status,
                component=component,
                details=details or {},
            )
        )

    def _emit_failure(
        self,
        *,
        component: str,
        error: Exception,
    ) -> None:
        self._emit(
            status="failed",
            component=component,
            details={"error_type": type(error).__name__},
        )
