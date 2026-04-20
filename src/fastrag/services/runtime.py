from collections.abc import Sequence
from typing import Any

from fastrag.errors import PipelineStageError
from fastrag.observability import ObservabilityHub, PipelineEvent
from fastrag.protocols import LLM, Chunker, Embedder, VectorStore
from fastrag.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    RAGResponse,
    SourceDocument,
)


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
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> IngestResponse:
        source_documents = self._build_source_documents(request)
        self._emit(
            operation="ingest",
            stage="chunk",
            status="started",
            component=type(chunker).__name__,
            details={"documents": len(source_documents)},
        )
        try:
            chunks = await chunker.chunk(source_documents)
            self._emit(
                operation="ingest",
                stage="chunk",
                status="succeeded",
                component=type(chunker).__name__,
                details={"documents": len(source_documents), "chunks": len(chunks)},
            )
        except Exception as exc:
            self._emit_failure(
                operation="ingest",
                stage="chunk",
                component=type(chunker).__name__,
                error=exc,
            )
            raise PipelineStageError(
                code="chunking_failed",
                message="Failed to chunk the ingest documents.",
                stage="chunk",
                details={"component": type(chunker).__name__},
            ) from exc

        self._emit(
            operation="ingest",
            stage="embed",
            status="started",
            component=type(embedder).__name__,
            details={"documents": len(source_documents), "chunks": len(chunks)},
        )
        try:
            embeddings = await embedder.embed([chunk.content for chunk in chunks])
            self._emit(
                operation="ingest",
                stage="embed",
                status="succeeded",
                component=type(embedder).__name__,
                details={"documents": len(source_documents), "chunks": len(chunks)},
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
            details={"documents": len(source_documents), "chunks": len(chunks)},
        )
        try:
            await vector_store.upsert(
                chunks=chunks,
                embeddings=embeddings,
                collection=request.collection,
                tenant_id=request.tenant_id,
            )
            self._emit(
                operation="ingest",
                stage="store",
                status="succeeded",
                component=type(vector_store).__name__,
                details={"documents": len(source_documents), "chunks": len(chunks)},
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
            ingested_documents=len(source_documents),
            collection=request.collection,
            tenant_id=request.tenant_id,
        )

    def _build_source_documents(self, request: IngestRequest) -> list[SourceDocument]:
        source_id = request.metadata.get("source_id")
        page_number = self._coerce_page_number(request.metadata.get("page_number"))
        source_documents: list[SourceDocument] = []
        for index, _document in enumerate(request.documents):
            metadata = dict(request.metadata)
            normalized_source_id = f"doc-{index}"
            if isinstance(source_id, str) and source_id.strip():
                normalized_source_id = source_id.strip()
                if len(request.documents) > 1:
                    normalized_source_id = f"{normalized_source_id}-{index}"
                metadata["source_id"] = normalized_source_id
            source_documents.append(
                SourceDocument(
                    source_id=normalized_source_id,
                    content=request.documents[index],
                    metadata=metadata,
                    page_number=page_number,
                )
            )

        return source_documents

    def build_source_documents_for_testing(
        self,
        request: IngestRequest,
    ) -> Sequence[SourceDocument]:
        """Visible wrapper for focused unit tests."""
        return self._build_source_documents(request)

    def _coerce_page_number(self, raw_page_number: Any) -> int | None:
        if isinstance(raw_page_number, int) and raw_page_number > 0:
            return raw_page_number
        return None

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
