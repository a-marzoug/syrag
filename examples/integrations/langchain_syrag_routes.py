import os
from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from syrag import (
    ChromaVectorStore,
    IngestRequest,
    OpenAIEmbedder,
    OpenAILLM,
    QueryRequest,
    RetrievedChunk,
    Settings,
    SyRAG,
)
from syrag.integrations.langchain import LangChainRetrieverStrategy, LangChainTextChunker
from syrag.protocols import EmbeddingVector, Filters, VectorStore
from syrag.schemas import DocumentChunk

SUPPORT_COLLECTION = "support"
QDRANT_COLLECTION = "support_docs"

syrag = SyRAG(
    title="LangChain-backed SyRAG Routes",
    version="0.2.0",
    description="SyRAG routes using LangChain splitter and retriever adapters.",
    settings=Settings(),
)

syrag_embedder = OpenAIEmbedder(
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-3-small",
)
syrag_llm = OpenAILLM(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4.1-mini",
)

langchain_embeddings = OpenAIEmbeddings(
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-3-small",
)
qdrant_client = QdrantClient(path=".syrag/langchain-route-qdrant")
vector_size = len(langchain_embeddings.embed_query("dimension probe"))

if not qdrant_client.collection_exists(QDRANT_COLLECTION):
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

langchain_vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
    embedding=langchain_embeddings,
)
langchain_vector_store.add_documents(
    [
        Document(
            page_content="SyRAG exposes typed ingest and query routes for RAG services.",
            metadata={
                "source_id": "overview",
                "chunk_id": "overview-0",
                "topic": "api",
            },
        ),
        Document(
            page_content="LangChain retrievers can be adapted into SyRAG query routes.",
            metadata={
                "source_id": "langchain",
                "chunk_id": "langchain-0",
                "topic": "integrations",
            },
        ),
    ]
)
langchain_retriever = langchain_vector_store.as_retriever(search_kwargs={"k": 5})

langchain_chunker = LangChainTextChunker(
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
)
syrag_storage = ChromaVectorStore(
    path=Path(".syrag/langchain-route-chroma"),
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
        raise RuntimeError("LangChainRetrieverStrategy owns query-time retrieval.")

    async def query(
        self,
        *,
        query_embedding: EmbeddingVector,
        top_k: int,
        collection: str | None = None,
        tenant_id: str | None = None,
        filters: Filters | None = None,
    ) -> list[RetrievedChunk]:
        raise RuntimeError("LangChainRetrieverStrategy owns query-time retrieval.")


@syrag.ingest(
    "/ingest",
    chunker=langchain_chunker,
    embedder=syrag_embedder,
    vector_store=syrag_storage,
)
async def ingest(request: IngestRequest) -> IngestRequest:
    return request.model_copy(
        update={
            "collection": request.collection or SUPPORT_COLLECTION,
            "metadata": {
                "source": "langchain-text-splitter",
                **request.metadata,
            },
        }
    )


@syrag.query(
    "/query",
    embedder=syrag_embedder,
    vector_store=RetrieverOwnedVectorStore(),
    llm=syrag_llm,
    retrieval_strategy=LangChainRetrieverStrategy(retriever=langchain_retriever),
)
async def query(request: QueryRequest) -> QueryRequest:
    return request.model_copy(
        update={
            "collection": request.collection or SUPPORT_COLLECTION,
            "top_k": min(request.top_k, 5),
        }
    )


api = syrag.api
