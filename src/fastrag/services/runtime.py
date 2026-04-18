from collections.abc import Sequence
from typing import Any

from fastrag.errors import PipelineStageError
from fastrag.observability import ObservabilityHub, PipelineEvent
from fastrag.protocols import LLM, Embedder, VectorStore
from fastrag.schemas import IngestRequest, IngestResponse, QueryRequest, RAGResponse


class PipelineService:
    """Internal orchestrator for FastRAG route execution."""

    def __init__(self, observability: ObservabilityHub | None = None) -> None:
        self.observability = observability or ObservabilityHub()

    async def run_query(
        self,
        *,
        request: QueryRequest,
        embedder: Embedder,
        vector_store: VectorStore,
        llm: LLM,
    ) -> RAGResponse:
        self._emit(
            operation="query",
            stage="embed",
            status="started",
            component=type(embedder).__name__,
        )
        try:
            query_embedding = (await embedder.embed([request.query]))[0]
            self._emit(
                operation="query",
                stage="embed",
                status="succeeded",
                component=type(embedder).__name__,
            )
        except Exception as exc:
            self._emit_failure(
                operation="query",
                stage="embed",
                component=type(embedder).__name__,
                error=exc,
            )
            raise PipelineStageError(
                code="embedding_failed",
                message="Failed to embed the query.",
                stage="embed",
                details={"component": type(embedder).__name__},
            ) from exc

        self._emit(
            operation="query",
            stage="retrieve",
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
                operation="query",
                stage="retrieve",
                status="succeeded",
                component=type(vector_store).__name__,
                details={"results": len(context)},
            )
        except Exception as exc:
            self._emit_failure(
                operation="query",
                stage="retrieve",
                component=type(vector_store).__name__,
                error=exc,
            )
            raise PipelineStageError(
                code="retrieval_failed",
                message="Failed to retrieve supporting documents.",
                stage="retrieve",
                details={"component": type(vector_store).__name__},
            ) from exc

        self._emit(
            operation="query",
            stage="generate",
            status="started",
            component=type(llm).__name__,
        )
        try:
            response = await llm.generate(query=request, context=context)
            self._emit(
                operation="query",
                stage="generate",
                status="succeeded",
                component=type(llm).__name__,
                details={"citations": len(response.citations)},
            )
            return response
        except Exception as exc:
            self._emit_failure(
                operation="query",
                stage="generate",
                component=type(llm).__name__,
                error=exc,
            )
            raise PipelineStageError(
                code="generation_failed",
                message="Failed to generate a grounded response.",
                stage="generate",
                details={"component": type(llm).__name__},
            ) from exc

    async def run_ingest(
        self,
        *,
        request: IngestRequest,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> IngestResponse:
        self._emit(
            operation="ingest",
            stage="embed",
            status="started",
            component=type(embedder).__name__,
            details={"documents": len(request.documents)},
        )
        try:
            embeddings = await embedder.embed(request.documents)
            self._emit(
                operation="ingest",
                stage="embed",
                status="succeeded",
                component=type(embedder).__name__,
                details={"documents": len(request.documents)},
            )
        except Exception as exc:
            self._emit_failure(
                operation="ingest",
                stage="embed",
                component=type(embedder).__name__,
                error=exc,
            )
            raise PipelineStageError(
                code="embedding_failed",
                message="Failed to embed the ingest documents.",
                stage="embed",
                details={"component": type(embedder).__name__},
            ) from exc

        self._emit(
            operation="ingest",
            stage="store",
            status="started",
            component=type(vector_store).__name__,
            details={"documents": len(request.documents)},
        )
        try:
            await vector_store.upsert(
                documents=request.documents,
                embeddings=embeddings,
                collection=request.collection,
                tenant_id=request.tenant_id,
                metadata=self._expand_ingest_metadata(request),
            )
            self._emit(
                operation="ingest",
                stage="store",
                status="succeeded",
                component=type(vector_store).__name__,
                details={"documents": len(request.documents)},
            )
        except Exception as exc:
            self._emit_failure(
                operation="ingest",
                stage="store",
                component=type(vector_store).__name__,
                error=exc,
            )
            raise PipelineStageError(
                code="storage_failed",
                message="Failed to persist embedded documents.",
                stage="store",
                details={"component": type(vector_store).__name__},
            ) from exc

        return IngestResponse(
            status="completed",
            ingested_documents=len(request.documents),
            collection=request.collection,
            tenant_id=request.tenant_id,
        )

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

    def expand_ingest_metadata_for_testing(
        self,
        request: IngestRequest,
    ) -> Sequence[dict[str, Any]]:
        """Visible wrapper for focused unit tests."""
        return self._expand_ingest_metadata(request)

    def _emit(
        self,
        *,
        operation: str,
        stage: str,
        status: str,
        component: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.observability.emit(
            PipelineEvent(
                operation=operation,
                stage=stage,
                status=status,
                component=component,
                details=details or {},
            )
        )

    def _emit_failure(
        self,
        *,
        operation: str,
        stage: str,
        component: str,
        error: Exception,
    ) -> None:
        self._emit(
            operation=operation,
            stage=stage,
            status="failed",
            component=component,
            details={"error_type": type(error).__name__},
        )
