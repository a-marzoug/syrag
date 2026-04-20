from typing import Any

from fastrag.errors import PipelineStageError
from fastrag.observability import ObservabilityHub, PipelineEvent
from fastrag.protocols import LLM, Chunker, Embedder, VectorStore
from fastrag.schemas import IngestRequest, IngestResponse, QueryRequest, RAGResponse
from fastrag.services.ingest import DefaultIngestionPipeline, IngestionPipeline


class PipelineService:
    """Internal orchestrator for FastRAG route execution."""

    def __init__(
        self,
        observability: ObservabilityHub | None = None,
        ingestion_pipeline: IngestionPipeline | None = None,
    ) -> None:
        self.observability = observability or ObservabilityHub()
        self.ingestion_pipeline = ingestion_pipeline or DefaultIngestionPipeline(
            observability=self.observability
        )

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
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> IngestResponse:
        return await self.ingestion_pipeline.run(
            request=request,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )

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
