import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from syrag.integrations.llamaindex import LlamaIndexNodeChunker, LlamaIndexRetrieverStrategy
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
