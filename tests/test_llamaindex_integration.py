import pytest
from llama_index.core.schema import TextNode

from syrag.integrations.llamaindex import LlamaIndexNodeChunker
from syrag.protocols import Chunker
from syrag.schemas import SourceDocument


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
