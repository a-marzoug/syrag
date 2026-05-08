import httpx
import pytest
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode

from syrag.integrations.llamaindex import (
    LlamaIndexNodeChunker,
    LlamaIndexRetrieverStrategy,
    SyRAGQueryEngine,
)
from syrag.protocols import Chunker
from syrag.schemas import QueryRequest, SourceDocument
from syrag.services import RetrievalStrategy


class FakeLlamaIndexNodeParser:
    def __init__(self) -> None:
        self.documents: list[object] = []
        self.show_progress_values: list[bool] = []

    def get_nodes_from_documents(
        self,
        documents: list[object],
        *,
        show_progress: bool = True,
    ) -> list[TextNode]:
        self.documents = documents
        self.show_progress_values.append(show_progress)
        document = documents[0]
        assert hasattr(document, "metadata")
        return [
            TextNode(
                id_="guide-node-0",
                text="First parsed node.",
                metadata={
                    **document.metadata,
                    "topic": "integrations",
                },
            ),
            TextNode(
                id_="guide-node-1",
                text="Second parsed node.",
                metadata=document.metadata,
            ),
        ]


class InvalidLlamaIndexNodeParser:
    pass


class FakeLlamaIndexRetriever:
    def __init__(self, nodes: list[NodeWithScore]) -> None:
        self.nodes = nodes
        self.calls: list[str] = []

    async def aretrieve(self, query: str) -> list[NodeWithScore]:
        self.calls.append(query)
        return self.nodes


class FakeSyncLlamaIndexRetriever:
    def __init__(self, nodes: list[NodeWithScore]) -> None:
        self.nodes = nodes
        self.calls: list[str] = []

    def retrieve(self, query: str) -> list[NodeWithScore]:
        self.calls.append(query)
        return self.nodes


class InvalidLlamaIndexRetriever:
    pass


@pytest.mark.asyncio
async def test_llamaindex_node_chunker_adapts_node_parser_output() -> None:
    node_parser = FakeLlamaIndexNodeParser()
    chunker = LlamaIndexNodeChunker(node_parser=node_parser)

    chunks = await chunker.chunk(
        [
            SourceDocument(
                source_id="guide",
                content="SyRAG integrates LlamaIndex node parsers.",
                metadata={"category": "docs"},
                page_number=5,
            )
        ]
    )

    assert isinstance(chunker, Chunker)
    assert node_parser.show_progress_values == [False]
    document = node_parser.documents[0]
    assert hasattr(document, "text")
    assert document.text == "SyRAG integrates LlamaIndex node parsers."
    assert chunks[0].chunk_id == "guide-node-0"
    assert chunks[0].source_id == "guide"
    assert chunks[0].content == "First parsed node."
    assert chunks[0].metadata == {"category": "docs", "topic": "integrations"}
    assert chunks[0].page_number == 5
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_id == "guide-node-1"
    assert chunks[1].source_id == "guide"
    assert chunks[1].content == "Second parsed node."
    assert chunks[1].metadata == {"category": "docs"}
    assert chunks[1].page_number == 5
    assert chunks[1].chunk_index == 1


def test_llamaindex_node_chunker_requires_get_nodes_from_documents() -> None:
    with pytest.raises(
        TypeError,
        match=r"node_parser must expose a callable get_nodes_from_documents",
    ):
        LlamaIndexNodeChunker(node_parser=InvalidLlamaIndexNodeParser())


@pytest.mark.asyncio
async def test_llamaindex_retriever_strategy_maps_nodes_to_retrieved_chunks() -> None:
    retriever = FakeLlamaIndexRetriever(
        nodes=[
            NodeWithScore(
                node=TextNode(
                    id_="overview-node-2",
                    text="SyRAG adapts LlamaIndex retrievers.",
                    metadata={
                        "source_id": "overview",
                        "chunk_index": 2,
                        "page_number": 4,
                        "topic": "framework",
                    },
                ),
                score=0.88,
            ),
            NodeWithScore(
                node=TextNode(
                    id_="overflow-node",
                    text="Extra result should be trimmed.",
                    metadata={"source_id": "overflow"},
                ),
                score=0.5,
            ),
        ]
    )
    strategy = LlamaIndexRetrieverStrategy(retriever=retriever)

    chunks = await strategy.retrieve(
        request=QueryRequest(query="What does SyRAG adapt?", top_k=1),
        query_embedding=[1.0, 0.0],
        vector_store=object(),  # type: ignore[arg-type]
    )

    assert isinstance(strategy, RetrievalStrategy)
    assert retriever.calls == ["What does SyRAG adapt?"]
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "overview-node-2"
    assert chunks[0].source_id == "overview"
    assert chunks[0].content == "SyRAG adapts LlamaIndex retrievers."
    assert chunks[0].score == 0.88
    assert chunks[0].metadata == {
        "source_id": "overview",
        "chunk_index": 2,
        "page_number": 4,
        "topic": "framework",
    }
    assert chunks[0].page_number == 4
    assert chunks[0].chunk_index == 2


@pytest.mark.asyncio
async def test_llamaindex_retriever_strategy_supports_sync_retrievers() -> None:
    retriever = FakeSyncLlamaIndexRetriever(
        nodes=[
            NodeWithScore(
                node=TextNode(
                    id_="sync-node",
                    text="SyRAG can call sync LlamaIndex retrievers.",
                    metadata={"source_id": "sync-doc"},
                ),
                score=0.77,
            )
        ]
    )
    strategy = LlamaIndexRetrieverStrategy(retriever=retriever)

    chunks = await strategy.retrieve(
        request=QueryRequest(query="sync?", top_k=5),
        query_embedding=[1.0],
        vector_store=object(),  # type: ignore[arg-type]
    )

    assert retriever.calls == ["sync?"]
    assert chunks[0].chunk_id == "sync-node"
    assert chunks[0].source_id == "sync-doc"
    assert chunks[0].score == 0.77


def test_llamaindex_retriever_strategy_requires_retrieve_or_aretrieve() -> None:
    with pytest.raises(
        TypeError,
        match=r"retriever must expose callable retrieve\(query\) or aretrieve\(query\)",
    ):
        LlamaIndexRetrieverStrategy(retriever=InvalidLlamaIndexRetriever())


def test_syrag_query_engine_calls_query_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "answer": "SyRAG can serve as a LlamaIndex query engine.",
                "citations": [
                    {
                        "source_id": "overview",
                        "score": 0.93,
                        "snippet": "SyRAG owns the service boundary.",
                        "page_number": 2,
                    }
                ],
                "usage": {"total_tokens": 31},
            },
        )

    query_engine = SyRAGQueryEngine(
        base_url="https://syrag.example",
        top_k=2,
        collection="docs",
        tenant_id="tenant-a",
        filters={"topic": "llamaindex"},
        headers={"x-tenant-id": "tenant-a"},
        transport=httpx.MockTransport(handler),
    )

    response = query_engine.query("What can SyRAG serve as?")

    assert isinstance(query_engine, BaseQueryEngine)
    assert str(response) == "SyRAG can serve as a LlamaIndex query engine."
    assert response.source_nodes[0].node.get_content() == (
        "SyRAG owns the service boundary."
    )
    assert response.source_nodes[0].score == 0.93
    assert response.metadata == {
        "usage": {"total_tokens": 31},
        "citations": [
            {
                "source_id": "overview",
                "score": 0.93,
                "snippet": "SyRAG owns the service boundary.",
                "page_number": 2,
            }
        ],
    }
    assert requests[0].url == "https://syrag.example/query"
    assert requests[0].headers["x-tenant-id"] == "tenant-a"
    assert requests[0].read() == (
        b'{"query":"What can SyRAG serve as?",'
        b'"top_k":2,'
        b'"filters":{"topic":"llamaindex"},'
        b'"collection":"docs",'
        b'"tenant_id":"tenant-a"}'
    )


@pytest.mark.asyncio
async def test_syrag_query_engine_supports_async_queries() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://syrag.example/api/query"
        return httpx.Response(
            status_code=200,
            json={
                "answer": "Async LlamaIndex can call SyRAG.",
                "citations": [],
                "usage": {},
            },
        )

    query_engine = SyRAGQueryEngine(
        base_url="https://syrag.example/api",
        transport=httpx.MockTransport(handler),
    )

    response = await query_engine.aquery("Can async LlamaIndex call SyRAG?")

    assert str(response) == "Async LlamaIndex can call SyRAG."
