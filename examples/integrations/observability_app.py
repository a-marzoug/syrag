import logging
import os
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except ModuleNotFoundError:
    OTLPSpanExporter = None

from syrag import (
    ChromaVectorStore,
    IngestRequest,
    JSONLogFormatter,
    OpenAIEmbedder,
    OpenAILLM,
    QueryRequest,
    Settings,
    SyRAG,
)

SUPPORT_COLLECTION = "support"


def configure_json_logger() -> logging.Logger:
    logger = logging.getLogger("syrag")
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setFormatter(JSONLogFormatter())
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def configure_otel_tracer() -> trace.Tracer:
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": os.environ.get("OTEL_SERVICE_NAME", "syrag-support"),
                "deployment.environment": os.environ.get("SYRAG_ENVIRONMENT", "development"),
            }
        )
    )

    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") and OTLPSpanExporter is not None:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    return trace.get_tracer("syrag.app")


syrag = SyRAG(
    title="Observable Support Bot",
    version="0.1.0",
    description="SyRAG app with JSON logs and OpenTelemetry tracing.",
    settings=Settings(environment=os.environ.get("SYRAG_ENVIRONMENT", "development")),
)
syrag.configure_logging(logger=configure_json_logger())
syrag.configure_tracing(tracer=configure_otel_tracer())

embedder = OpenAIEmbedder(
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-3-small",
)
vector_store = ChromaVectorStore(
    path=Path(".syrag/observability-chroma"),
    collection_name="support_docs",
)
llm = OpenAILLM(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4.1-mini",
)


@syrag.ingest("/ingest", embedder=embedder, vector_store=vector_store)
async def ingest(request: IngestRequest) -> IngestRequest:
    return request.model_copy(
        update={
            "collection": request.collection or SUPPORT_COLLECTION,
            "metadata": {
                "source": "observable-api",
                **request.metadata,
            },
        }
    )


@syrag.query("/query", embedder=embedder, vector_store=vector_store, llm=llm)
async def query(request: QueryRequest) -> QueryRequest:
    return request.model_copy(
        update={
            "collection": request.collection or SUPPORT_COLLECTION,
            "top_k": min(request.top_k, 5),
        }
    )


api = syrag.api
