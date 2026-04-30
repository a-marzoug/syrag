import os

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
)
Settings.llm = OpenAI(
    model="gpt-4.1-mini",
    api_key=os.environ["OPENAI_API_KEY"],
)

client = QdrantClient(path=".syrag/llamaindex-qdrant")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="support_docs",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = [
    Document(
        text="SyRAG exposes typed ingest and query routes for RAG services.",
        metadata={"source": "overview", "topic": "api"},
    ),
    Document(
        text="Qdrant stores vectors and payloads for semantic retrieval.",
        metadata={"source": "qdrant", "topic": "vector-store"},
    ),
]

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
query_engine = index.as_query_engine(similarity_top_k=3)

if __name__ == "__main__":
    response = query_engine.query("What does SyRAG expose?")
    print(response)
