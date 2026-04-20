from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from fastrag.errors import PipelineStageError
from fastrag.observability import ObservabilityHub, PipelineEvent
from fastrag.protocols import Chunker, Embedder, VectorStore
from fastrag.schemas import IngestRequest, IngestResponse, SourceDocument


class IngestionPipeline(Protocol):
    """Contract for framework-managed ingestion execution."""

    async def run(
        self,
        *,
        request: IngestRequest,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> IngestResponse:
        """Execute normalization, chunking, embedding, and storage for an ingest request."""


class DefaultIngestionPipeline:
    """Default ingestion pipeline used by the framework runtime."""

    def __init__(self, observability: ObservabilityHub) -> None:
        self.observability = observability

    async def run(
        self,
        *,
        request: IngestRequest,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> IngestResponse:
        source_documents = self._build_source_documents(request)
        self._emit(
            stage="chunk",
            status="started",
            component=type(chunker).__name__,
            details={"documents": len(source_documents)},
        )
        try:
            chunks = await chunker.chunk(source_documents)
            self._emit(
                stage="chunk",
                status="succeeded",
                component=type(chunker).__name__,
                details={"documents": len(source_documents), "chunks": len(chunks)},
            )
        except Exception as exc:
            self._emit_failure(
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
            stage="embed",
            status="started",
            component=type(embedder).__name__,
            details={"documents": len(source_documents), "chunks": len(chunks)},
        )
        try:
            embeddings = await embedder.embed([chunk.content for chunk in chunks])
            self._emit(
                stage="embed",
                status="succeeded",
                component=type(embedder).__name__,
                details={"documents": len(source_documents), "chunks": len(chunks)},
            )
        except Exception as exc:
            self._emit_failure(
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
                stage="store",
                status="succeeded",
                component=type(vector_store).__name__,
                details={"documents": len(source_documents), "chunks": len(chunks)},
            )
        except Exception as exc:
            self._emit_failure(
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

    def build_source_documents_for_testing(
        self,
        request: IngestRequest,
    ) -> Sequence[SourceDocument]:
        """Visible wrapper for focused unit tests."""
        return self._build_source_documents(request)

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

    def _coerce_page_number(self, raw_page_number: Any) -> int | None:
        if isinstance(raw_page_number, int) and raw_page_number > 0:
            return raw_page_number
        return None

    def _emit(
        self,
        *,
        stage: str,
        status: str,
        component: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.observability.emit(
            PipelineEvent(
                operation="ingest",
                stage=stage,
                status=status,
                component=component,
                details=details or {},
            )
        )

    def _emit_failure(
        self,
        *,
        stage: str,
        component: str,
        error: Exception,
    ) -> None:
        self._emit(
            stage=stage,
            status="failed",
            component=component,
            details={"error_type": type(error).__name__},
        )
