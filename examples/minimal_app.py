from syrag import (
    IngestRequest,
    InMemoryEmbedder,
    InMemoryLLM,
    InMemoryVectorStore,
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

syrag.register_embedder("default", InMemoryEmbedder())
syrag.register_vector_store("default", InMemoryVectorStore())
syrag.register_llm("default", InMemoryLLM())
syrag.configure_defaults(embedder="default", vector_store="default", llm="default")


@syrag.ingest("/ingest")
async def ingest(request: IngestRequest) -> IngestRequest:
    return request


@syrag.query("/query")
async def query(request: QueryRequest) -> QueryRequest:
    return request


api = syrag.api
