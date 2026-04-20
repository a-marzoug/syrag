from typing import Any

from fastrag.errors import PipelineStageError
from fastrag.observability import ObservabilityHub, PipelineEvent
from fastrag.protocols import LLM, Chunker, Embedder, PromptAssembler, VectorStore
from fastrag.schemas import IngestRequest, IngestResponse, QueryRequest, RAGResponse
from fastrag.services.assembly import DefaultPromptAssembler
from fastrag.services.ingest import DefaultIngestionPipeline, IngestionPipeline
from fastrag.services.retrieval import DefaultRetrievalStrategy, RetrievalStrategy


class PipelineService:
    """Internal orchestrator for FastRAG route execution."""

    def __init__(
        self,
        observability: ObservabilityHub | None = None,
        ingestion_pipeline: IngestionPipeline | None = None,
        retrieval_strategy: RetrievalStrategy | None = None,
        prompt_assembler: PromptAssembler | None = None,
    ) -> None:
        self.observability = observability or ObservabilityHub()
        self.ingestion_pipeline = ingestion_pipeline or DefaultIngestionPipeline(
            observability=self.observability
        )
        self.retrieval_strategy = retrieval_strategy or DefaultRetrievalStrategy(
            observability=self.observability
        )
        self.prompt_assembler = prompt_assembler or DefaultPromptAssembler()

    async def run_query(
        self,
        *,
        request: QueryRequest,
        embedder: Embedder,
        vector_store: VectorStore,
        llm: LLM,
        retrieval_strategy: RetrievalStrategy | None = None,
        prompt_assembler: PromptAssembler | None = None,
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

        context = await (retrieval_strategy or self.retrieval_strategy).retrieve(
            request=request,
            query_embedding=query_embedding,
            vector_store=vector_store,
        )

        assembler = prompt_assembler or self.prompt_assembler
        self._emit(
            operation="query",
            stage="assemble",
            status="started",
            component=type(assembler).__name__,
        )
        try:
            assembled_prompt = await assembler.assemble(query=request, context=context)
            self._emit(
                operation="query",
                stage="assemble",
                status="succeeded",
                component=type(assembler).__name__,
                details={"context_chunks": len(assembled_prompt.context)},
            )
        except Exception as exc:
            self._emit_failure(
                operation="query",
                stage="assemble",
                component=type(assembler).__name__,
                error=exc,
            )
            raise PipelineStageError(
                code="assembly_failed",
                message="Failed to assemble the grounded prompt.",
                stage="assemble",
                details={"component": type(assembler).__name__},
            ) from exc

        self._emit(
            operation="query",
            stage="generate",
            status="started",
            component=type(llm).__name__,
        )
        try:
            response = await llm.generate(prompt=assembled_prompt)
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
