import pytest
from pydantic import ValidationError

from syrag.schemas import Citation, DocumentChunk, IngestRequest, QueryRequest, RAGResponse


def test_ingest_request_requires_at_least_one_document() -> None:
    with pytest.raises(ValidationError):
        IngestRequest(documents=[])


def test_query_request_rejects_blank_queries() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="   ")


def test_citation_score_must_stay_in_normalized_range() -> None:
    with pytest.raises(ValidationError):
        Citation(source_id="doc-1", score=1.5, snippet="Relevant excerpt.")


def test_rag_response_preserves_typed_citations() -> None:
    response = RAGResponse(
        answer="SyRAG wraps FastAPI with framework-oriented primitives.",
        citations=[
            Citation(
                source_id="prd",
                score=0.98,
                snippet="Ship a fully observable, multi-tenant RAG service.",
                page_number=1,
            )
        ],
        usage={"prompt_tokens": 128, "completion_tokens": 42},
    )

    assert response.citations[0].source_id == "prd"
    assert response.usage["completion_tokens"] == 42


def test_document_chunk_requires_non_negative_chunk_index() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="chunk-1",
            source_id="doc-1",
            content="Chunk content",
            chunk_index=-1,
        )
