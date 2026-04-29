from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SyRAGSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class IngestRequest(SyRAGSchema):
    documents: list[str] = Field(min_length=1)
    collection: str | None = Field(default=None)
    tenant_id: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(SyRAGSchema):
    status: Literal["accepted", "completed"]
    ingested_documents: int = Field(ge=0)
    collection: str | None = Field(default=None)
    tenant_id: str | None = Field(default=None)


class QueryRequest(SyRAGSchema):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    collection: str | None = Field(default=None)
    tenant_id: str | None = Field(default=None)
    filters: dict[str, Any] = Field(default_factory=dict)


class RequestContext(SyRAGSchema):
    request_id: str | None = Field(default=None)
    tenant_id: str | None = Field(default=None)
    subject_id: str | None = Field(default=None)
    auth_scheme: str | None = Field(default=None)
    scopes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(SyRAGSchema):
    source_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    snippet: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)


class SourceDocument(SyRAGSchema):
    source_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    page_number: int | None = Field(default=None, ge=1)


class DocumentChunk(SyRAGSchema):
    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    page_number: int | None = Field(default=None, ge=1)
    chunk_index: int = Field(default=0, ge=0)


class RetrievedChunk(DocumentChunk):
    score: float = Field(ge=0.0, le=1.0)


class RetrievedDocument(RetrievedChunk):
    """Backward-compatible alias for retrieved chunk results."""


class AssembledPrompt(SyRAGSchema):
    query: QueryRequest
    context: list[RetrievedChunk] = Field(default_factory=list)
    prompt: str = Field(min_length=1)


class GenerationRequest(SyRAGSchema):
    query: QueryRequest
    context: list[RetrievedChunk] = Field(default_factory=list)
    prompt: str = Field(min_length=1)
    system_prompt: str | None = Field(default=None)
    require_citations: bool = Field(default=True)


class RAGResponse(SyRAGSchema):
    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


class ErrorDetail(SyRAGSchema):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(SyRAGSchema):
    error: ErrorDetail
