import httpx
import pytest
from langchain_core.documents import Document

from syrag.integrations.langchain import (
    LangChainRetrieverStrategy,
    LangChainTextChunker,
    SyRAGQueryToolInput,
    create_syrag_query_tool,
)
from syrag.protocols import Chunker
from syrag.schemas import QueryRequest, SourceDocument
from syrag.services import RetrievalStrategy


class FakeLangChainTextSplitter:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.calls: list[str] = []

    def split_text(self, text: str) -> list[str]:
        self.calls.append(text)
        return self.chunks


class InvalidLangChainTextSplitter:
    pass


class FakeLangChainRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls: list[str] = []

    async def ainvoke(self, query: str) -> list[Document]:
        self.calls.append(query)
        return self.documents


class FakeSyncLangChainRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        self.calls.append(query)
        return self.documents


class InvalidLangChainRetriever:
    pass


@pytest.mark.asyncio
async def test_langchain_text_chunker_adapts_split_text_output() -> None:
    text_splitter = FakeLangChainTextSplitter(
        chunks=[
            " First chunk. ",
            "Second chunk.",
            "",
        ]
    )
    chunker = LangChainTextChunker(text_splitter=text_splitter)

    chunks = await chunker.chunk(
        [
            SourceDocument(
                source_id="guide",
                content="SyRAG integrates LangChain text splitters.",
                metadata={"topic": "integrations"},
                page_number=3,
            )
        ]
    )

    assert isinstance(chunker, Chunker)
    assert text_splitter.calls == ["SyRAG integrates LangChain text splitters."]
    assert [chunk.chunk_id for chunk in chunks] == ["guide-chunk-0", "guide-chunk-1"]
    assert [chunk.content for chunk in chunks] == ["First chunk.", "Second chunk."]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert all(chunk.source_id == "guide" for chunk in chunks)
    assert all(chunk.metadata == {"topic": "integrations"} for chunk in chunks)
    assert all(chunk.page_number == 3 for chunk in chunks)


def test_langchain_text_chunker_requires_split_text() -> None:
    with pytest.raises(
        TypeError,
        match=r"text_splitter must expose a callable split_text\(text\) method",
    ):
        LangChainTextChunker(text_splitter=InvalidLangChainTextSplitter())


@pytest.mark.asyncio
async def test_langchain_text_chunker_rejects_non_string_chunks() -> None:
    text_splitter = FakeLangChainTextSplitter(chunks=["valid"])
    text_splitter.chunks = ["valid", object()]  # type: ignore[list-item]
    chunker = LangChainTextChunker(text_splitter=text_splitter)

    with pytest.raises(TypeError, match="must return only strings"):
        await chunker.chunk(
            [
                SourceDocument(
                    source_id="guide",
                    content="SyRAG integrates LangChain.",
                )
            ]
        )


@pytest.mark.asyncio
async def test_langchain_retriever_strategy_maps_documents_to_retrieved_chunks() -> None:
    retriever = FakeLangChainRetriever(
        documents=[
            Document(
                page_content="SyRAG wraps RAG services.",
                metadata={
                    "source_id": "overview",
                    "chunk_id": "overview-chunk-2",
                    "chunk_index": 2,
                    "page_number": 4,
                    "score": 0.87,
                    "topic": "framework",
                },
            ),
            Document(
                page_content="Extra result should be trimmed.",
                metadata={"source": "overflow"},
            ),
        ]
    )
    strategy = LangChainRetrieverStrategy(retriever=retriever)

    chunks = await strategy.retrieve(
        request=QueryRequest(query="What does SyRAG wrap?", top_k=1),
        query_embedding=[1.0, 0.0],
        vector_store=object(),  # type: ignore[arg-type]
    )

    assert isinstance(strategy, RetrievalStrategy)
    assert retriever.calls == ["What does SyRAG wrap?"]
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "overview-chunk-2"
    assert chunks[0].source_id == "overview"
    assert chunks[0].content == "SyRAG wraps RAG services."
    assert chunks[0].score == 0.87
    assert chunks[0].metadata["topic"] == "framework"
    assert chunks[0].page_number == 4
    assert chunks[0].chunk_index == 2


@pytest.mark.asyncio
async def test_langchain_retriever_strategy_supports_sync_retrievers() -> None:
    retriever = FakeSyncLangChainRetriever(
        documents=[
            Document(
                page_content="SyRAG can call sync LangChain retrievers.",
                metadata={"source": "sync-doc", "relevance_score": 0.75},
            )
        ]
    )
    strategy = LangChainRetrieverStrategy(retriever=retriever)

    chunks = await strategy.retrieve(
        request=QueryRequest(query="sync?", top_k=5),
        query_embedding=[1.0],
        vector_store=object(),  # type: ignore[arg-type]
    )

    assert retriever.calls == ["sync?"]
    assert chunks[0].source_id == "sync-doc"
    assert chunks[0].chunk_id == "sync-doc-chunk-0"
    assert chunks[0].score == 0.75


def test_langchain_retriever_strategy_requires_invoke_or_ainvoke() -> None:
    with pytest.raises(
        TypeError,
        match=r"retriever must expose callable invoke\(query\) or ainvoke\(query\) methods",
    ):
        LangChainRetrieverStrategy(retriever=InvalidLangChainRetriever())


def test_create_syrag_query_tool_calls_query_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "answer": "SyRAG gives agents a service boundary.",
                "citations": [
                    {
                        "source_id": "overview",
                        "score": 0.91,
                        "snippet": "SyRAG owns the RAG service boundary.",
                        "page_number": 2,
                    }
                ],
                "usage": {"total_tokens": 42},
            },
        )

    tool = create_syrag_query_tool(
        base_url="https://syrag.example",
        headers={"x-tenant-id": "tenant-a"},
        transport=httpx.MockTransport(handler),
    )

    result = tool.invoke(
        {
            "query": "What does SyRAG give agents?",
            "top_k": 2,
            "collection": "docs",
            "tenant_id": "tenant-a",
            "filters": {"topic": "agents"},
        }
    )

    assert tool.name == "query_syrag"
    assert tool.args_schema is SyRAGQueryToolInput
    assert "SyRAG gives agents a service boundary." in result
    assert "overview, page 2 (score: 0.91)" in result
    assert len(requests) == 1
    assert requests[0].url == "https://syrag.example/query"
    assert requests[0].headers["x-tenant-id"] == "tenant-a"
    assert requests[0].read() == (
        b'{"query":"What does SyRAG give agents?",'
        b'"top_k":2,'
        b'"filters":{"topic":"agents"},'
        b'"collection":"docs",'
        b'"tenant_id":"tenant-a"}'
    )


@pytest.mark.asyncio
async def test_create_syrag_query_tool_supports_async_invocation() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://syrag.example/api/query"
        return httpx.Response(
            status_code=200,
            json={
                "answer": "Async agents can call SyRAG.",
                "citations": [],
                "usage": {},
            },
        )

    tool = create_syrag_query_tool(
        base_url="https://syrag.example/api",
        transport=httpx.MockTransport(handler),
    )

    result = await tool.ainvoke({"query": "Can async agents call SyRAG?"})

    assert result == "Async agents can call SyRAG."
