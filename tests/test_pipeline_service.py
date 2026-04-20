import pytest

from fastrag.providers import InMemoryEmbedder, InMemoryLLM, InMemoryVectorStore, PassThroughChunker
from fastrag.schemas import IngestRequest, QueryRequest
from fastrag.services import PipelineService


@pytest.mark.asyncio
async def test_pipeline_service_runs_ingest_and_query_flow() -> None:
    service = PipelineService()
    embedder = InMemoryEmbedder()
    vector_store = InMemoryVectorStore()
    llm = InMemoryLLM()

    ingest_response = await service.run_ingest(
        request=IngestRequest(
            documents=[
                "FastRAG is a production-first Python framework for RAG services.",
                "It emphasizes observability and type safety.",
            ],
            collection="overview",
            metadata={"source_id": "overview", "page_number": 1},
        ),
        chunker=PassThroughChunker(),
        embedder=embedder,
        vector_store=vector_store,
    )
    query_response = await service.run_query(
        request=QueryRequest(
            query="What does FastRAG emphasize?",
            collection="overview",
            top_k=2,
        ),
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
    )

    assert ingest_response.ingested_documents == 2
    assert query_response.citations[0].source_id == "overview-0"
    assert "FastRAG" in query_response.answer


def test_pipeline_service_builds_source_documents() -> None:
    service = PipelineService()
    request = IngestRequest(
        documents=["doc one", "doc two"],
        metadata={"source_id": "doc", "page_number": 1},
    )
    source_documents = service.build_source_documents_for_testing(request)

    assert source_documents[0].source_id == "doc-0"
    assert source_documents[1].source_id == "doc-1"
