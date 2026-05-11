import os
from collections.abc import Sequence
from pathlib import Path

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from syrag import (
    ChromaVectorStore,
    IngestRequest,
    OpenAIEmbedder,
    OpenAILLM,
    QueryRequest,
    RetrievedChunk,
    SyRAG,
)
from syrag import (
    Settings as SyRAGSettings,
)
from syrag.integrations.llamaindex import LlamaIndexNodeChunker, LlamaIndexRetrieverStrategy
from syrag.protocols import EmbeddingVector, Filters, VectorStore
from syrag.schemas import DocumentChunk

SUPPORT_COLLECTION = "support"
QDRANT_COLLECTION = "support_docs"

syrag = SyRAG(
    title="LlamaIndex-backed SyRAG Routes",
    version="0.2.0",
    description="SyRAG routes using LlamaIndex node parser and retriever adapters.",
    settings=SyRAGSettings(),
)

syrag_embedder = OpenAIEmbedder(
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-3-small",
)
syrag_llm = OpenAILLM(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4.1-mini",
)

Settings.embed_model = OpenAIEmbedding(
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-3-small",
)

qdrant_client = QdrantClient(path=".syrag/llamaindex-route-qdrant")
llamaindex_vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
)
storage_context = StorageContext.from_defaults(vector_store=llamaindex_vector_store)
llamaindex_index = VectorStoreIndex.from_documents(
    [
        Document(
            text="SyRAG exposes typed ingest and query routes for RAG services.",
            metadata={
                "source_id": "overview",
                "topic": "api",
            },
        ),
        Document(
            text="LlamaIndex retrievers can be adapted into SyRAG query routes.",
            metadata={
                "source_id": "llamaindex",
                "topic": "integrations",
            },
        ),
    ],
    storage_context=storage_context,
)
llamaindex_retriever = llamaindex_index.as_retriever(similarity_top_k=5)

llamaindex_chunker = LlamaIndexNodeChunker(
    node_parser=SentenceSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
)
syrag_storage = ChromaVectorStore(
    path=Path(".syrag/llamaindex-route-chroma"),
    collection_name="syrag_ingested_docs",
)


class RetrieverOwnedVectorStore(VectorStore):
    """Route contract placeholder when a framework retriever owns retrieval."""

    async def upsert(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[EmbeddingVector],
        collection: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        raise RuntimeError("LlamaIndexRetrieverStrategy owns query-time retrieval.")

    async def query(
        self,
        *,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str | None = None,
        tenant_id: str | None = None,
        filters: Filters | None = None,
    ) -> list[RetrievedChunk]:
        raise RuntimeError("LlamaIndexRetrieverStrategy owns query-time retrieval.")


@syrag.ingest(
    "/ingest",
    chunker=llamaindex_chunker,
    embedder=syrag_embedder,
    vector_store=syrag_storage,
)
async def ingest(request: IngestRequest) -> IngestRequest:
    return request.model_copy(
        update={
            "collection": request.collection or SUPPORT_COLLECTION,
            "metadata": {
                "source": "llamaindex-node-parser",
                **request.metadata,
            },
        }
    )


@syrag.query(
    "/query",
    embedder=syrag_embedder,
    vector_store=RetrieverOwnedVectorStore(),
    llm=syrag_llm,
    retrieval_strategy=LlamaIndexRetrieverStrategy(retriever=llamaindex_retriever),
)
async def query(request: QueryRequest) -> QueryRequest:
    return request.model_copy(
        update={
            "collection": request.collection or SUPPORT_COLLECTION,
            "top_k": min(request.top_k, 5),
        }
    )


api = syrag.api
