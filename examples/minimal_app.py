import os
from pathlib import Path

from syrag import (
    ChromaVectorStore,
    IngestRequest,
    OpenAIEmbedder,
    OpenAILLM,
    QueryRequest,
    Settings,
    SyRAG,
)

syrag = SyRAG(
    title="Support Bot",
    version="0.1.0",
    description="Minimal SyRAG app",
    settings=Settings(),
)

syrag.register_embedder(
    "default",
    OpenAIEmbedder(
        api_key=os.environ["OPENAI_API_KEY"],
        model="text-embedding-3-small",
    ),
)
syrag.register_vector_store(
    "default",
    ChromaVectorStore(path=Path(".syrag/chroma"), collection_name="support_docs"),
)
syrag.register_llm(
    "default",
    OpenAILLM(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4.1-mini"),
)
syrag.configure_defaults(embedder="default", vector_store="default", llm="default")


@syrag.ingest("/ingest")
async def ingest(request: IngestRequest) -> IngestRequest:
    return request


@syrag.query("/query")
async def query(request: QueryRequest) -> QueryRequest:
    return request


api = syrag.api
