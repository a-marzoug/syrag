from collections.abc import Sequence

import pytest

from syrag.errors import PipelineStageError
from syrag.observability import ObservabilityHub, PipelineEvent
from syrag.protocols import EmbeddingVector, Reranker, VectorStore
from syrag.schemas import QueryRequest, RetrievedChunk
from syrag.services import RerankingRetrievalStrategy, RetrievalStrategy


class StubRetrievalStrategy:
    def __init__(self, context: list[RetrievedChunk]) -> None:
        self.context = context
        self.calls: list[QueryRequest] = []

    async def retrieve(
        self,
        *,
        request: QueryRequest,
        query_embedding: EmbeddingVector,
        vector_store: VectorStore,
    ) -> list[RetrievedChunk]:
        del query_embedding, vector_store
        self.calls.append(request)
        return self.context


class ScoreReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[QueryRequest, list[RetrievedChunk]]] = []

    async def rerank(
        self,
        *,
        query: QueryRequest,
        context: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        self.calls.append((query, list(context)))
        return sorted(context, key=lambda chunk: chunk.score, reverse=True)


class FailingReranker:
    async def rerank(
        self,
        *,
        query: QueryRequest,
        context: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        del query, context
        msg = "reranker unavailable"
        raise RuntimeError(msg)


@pytest.mark.asyncio
async def test_reranking_retrieval_strategy_reranks_and_trims_context() -> None:
    context = [
        _chunk(chunk_id="low", score=0.1),
        _chunk(chunk_id="high", score=0.9),
        _chunk(chunk_id="middle", score=0.5),
    ]
    base_strategy = StubRetrievalStrategy(context=context)
    reranker = ScoreReranker()
    observability = ObservabilityHub()
    events: list[PipelineEvent] = []
    observability.add_listener(events.append)
    strategy = RerankingRetrievalStrategy(
        base_strategy=base_strategy,
        reranker=reranker,
        observability=observability,
    )
    request = QueryRequest(query="What is SyRAG?", top_k=2)

    results = await strategy.retrieve(
        request=request,
        query_embedding=[1.0],
        vector_store=object(),  # type: ignore[arg-type]
    )

    assert isinstance(strategy, RetrievalStrategy)
    assert isinstance(reranker, Reranker)
    assert base_strategy.calls == [request]
    assert reranker.calls == [(request, context)]
    assert [chunk.chunk_id for chunk in results] == ["high", "middle"]
    assert [(event.stage, event.status) for event in events] == [
        ("rerank", "started"),
        ("rerank", "succeeded"),
    ]
    assert events[0].details == {"candidates": 3}
    assert events[1].details == {"results": 2}


@pytest.mark.asyncio
async def test_reranking_retrieval_strategy_skips_reranker_for_empty_context() -> None:
    base_strategy = StubRetrievalStrategy(context=[])
    reranker = ScoreReranker()
    strategy = RerankingRetrievalStrategy(
        base_strategy=base_strategy,
        reranker=reranker,
    )

    results = await strategy.retrieve(
        request=QueryRequest(query="No context?"),
        query_embedding=[1.0],
        vector_store=object(),  # type: ignore[arg-type]
    )

    assert results == []
    assert reranker.calls == []


@pytest.mark.asyncio
async def test_reranking_retrieval_strategy_wraps_reranker_failures() -> None:
    observability = ObservabilityHub()
    events: list[PipelineEvent] = []
    observability.add_listener(events.append)
    strategy = RerankingRetrievalStrategy(
        base_strategy=StubRetrievalStrategy(context=[_chunk(chunk_id="candidate")]),
        reranker=FailingReranker(),
        observability=observability,
    )

    with pytest.raises(PipelineStageError, match="Failed to rerank retrieved context") as exc:
        await strategy.retrieve(
            request=QueryRequest(query="What fails?"),
            query_embedding=[1.0],
            vector_store=object(),  # type: ignore[arg-type]
        )

    assert exc.value.code == "reranking_failed"
    assert exc.value.stage == "rerank"
    assert [(event.stage, event.status) for event in events] == [
        ("rerank", "started"),
        ("rerank", "failed"),
    ]
    assert events[1].details == {"error_type": "RuntimeError"}


def _chunk(*, chunk_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id=f"{chunk_id}-source",
        content=f"{chunk_id} content",
        score=score,
        metadata={},
        chunk_index=0,
    )
