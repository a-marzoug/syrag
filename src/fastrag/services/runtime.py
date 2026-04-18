from collections.abc import Sequence
from typing import Any

from fastrag.errors import PipelineStageError
from fastrag.protocols import LLM, Embedder, VectorStore
from fastrag.schemas import IngestRequest, IngestResponse, QueryRequest, RAGResponse


class PipelineService:
    """Internal orchestrator for FastRAG route execution."""

    async def run_query(
        self,
        *,
        request: QueryRequest,
        embedder: Embedder,
        vector_store: VectorStore,
        llm: LLM,
    ) -> RAGResponse:
        try:
            query_embedding = (await embedder.embed([request.query]))[0]
        except Exception as exc:
            raise PipelineStageError(
                code="embedding_failed",
                message="Failed to embed the query.",
                stage="embed",
                details={"component": type(embedder).__name__},
            ) from exc

        try:
            context = await vector_store.query(
                query_embedding=query_embedding,
                top_k=request.top_k,
                collection=request.collection,
                tenant_id=request.tenant_id,
                filters=request.filters,
            )
        except Exception as exc:
            raise PipelineStageError(
                code="retrieval_failed",
                message="Failed to retrieve supporting documents.",
                stage="retrieve",
                details={"component": type(vector_store).__name__},
            ) from exc

        try:
            return await llm.generate(query=request, context=context)
        except Exception as exc:
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
        try:
            embeddings = await embedder.embed(request.documents)
        except Exception as exc:
            raise PipelineStageError(
                code="embedding_failed",
                message="Failed to embed the ingest documents.",
                stage="embed",
                details={"component": type(embedder).__name__},
            ) from exc

        try:
            await vector_store.upsert(
                documents=request.documents,
                embeddings=embeddings,
                collection=request.collection,
                tenant_id=request.tenant_id,
                metadata=self._expand_ingest_metadata(request),
            )
        except Exception as exc:
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
