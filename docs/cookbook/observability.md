# Observability

SyRAG emits request-level logs and pipeline-stage events from the framework runtime. Applications can route those events into JSON logs, OpenTelemetry spans, or both without adding logging code to every route handler.

The complete observability example is available at [`examples/integrations/observability_app.py`](../../examples/integrations/observability_app.py).

## JSON Logs And OpenTelemetry

Use this pattern when deploying SyRAG behind an API gateway, container platform, or tracing backend.

Install:

```bash
pip install "syrag[chroma,openai,server]" opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

Configure local console traces:

```bash
export OPENAI_API_KEY="sk-..."
export SYRAG_ENVIRONMENT="development"
export OTEL_SERVICE_NAME="syrag-support"
```

Or configure an OTLP HTTP collector:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
```

Run the app:

```bash
uvicorn examples.integrations.observability_app:api --reload
```

The app configures JSON logs with `StructuredLogging` through `configure_logging`:

```python
import logging
import os

from syrag import JSONLogFormatter


def configure_json_logger() -> logging.Logger:
    logger = logging.getLogger("syrag")
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setFormatter(JSONLogFormatter())
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


syrag.configure_logging(logger=configure_json_logger())
```

It configures OpenTelemetry tracing with an OTLP exporter when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, otherwise it prints spans to stdout:

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


provider = TracerProvider(
    resource=Resource.create(
        {
            "service.name": "syrag-support",
            "deployment.environment": "development",
        }
    )
)
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

syrag.configure_tracing(tracer=trace.get_tracer("syrag.app"))
```

## What Gets Logged

Every request emits one request log with fields such as:

```json
{
  "event": "syrag.request",
  "status": "completed",
  "method": "POST",
  "path": "/query",
  "request_id": "88b6f2d4-8f55-4f67-9e4a-ef1c9f7d4a3d",
  "tenant_id": "tenant-a",
  "subject_id": "api-key:tenant-a",
  "status_code": 200,
  "duration_ms": 183.42
}
```

Each pipeline stage emits a structured log event:

```json
{
  "event": "syrag.pipeline",
  "operation": "query",
  "stage": "retrieve",
  "status": "succeeded",
  "component": "ChromaVectorStore",
  "details": {
    "results": 3
  }
}
```

## What Gets Traced

SyRAG creates:

- a `syrag.request` span for the HTTP request
- `syrag.ingest.chunk`, `syrag.ingest.embed`, and `syrag.ingest.store` spans for ingest
- `syrag.query.embed`, `syrag.query.retrieve`, `syrag.query.assemble`, `syrag.query.policy`, and `syrag.query.generate` spans for query

Important span attributes include:

- `syrag.request_id`
- `syrag.tenant_id`
- `syrag.subject_id`
- `syrag.auth_scheme`
- `syrag.operation`
- `syrag.stage`
- `syrag.component`
- `syrag.detail.results`

## Production Notes

- Set a stable `service.name` so traces and logs group correctly in your backend.
- Forward stdout/stderr to your platform log collector; SyRAG logs are already structured JSON.
- Use OTLP in production instead of the console exporter.
- Treat request IDs as correlation IDs across logs, traces, and HTTP responses.
- Avoid logging raw prompts or documents by default; keep observability metadata useful but non-sensitive.
