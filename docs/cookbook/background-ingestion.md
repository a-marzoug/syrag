# Background Ingestion

SyRAG’s `@syrag.ingest(...)` decorator runs ingestion inline: normalize, chunk, embed, and store before returning the response. That is the right default for small batches and tests, but production document ingestion often needs a job boundary.

The complete background ingestion example is available at [`examples/integrations/background_ingestion_app.py`](../../examples/integrations/background_ingestion_app.py).

## In-Process Background Jobs

Use this pattern for local deployments, demos, and small internal services. For production durability, replace FastAPI `BackgroundTasks` and the in-memory job dictionary with a queue such as Celery, RQ, Dramatiq, Arq, or a managed cloud queue.

Install:

```bash
pip install "syrag[chroma,openai,server]"
```

Configure and run:

```bash
export OPENAI_API_KEY="sk-..."
uvicorn examples.integrations.background_ingestion_app:api --reload
```

The example reuses SyRAG components and calls the ingestion pipeline from a background task:

```python
from fastapi import BackgroundTasks, Response, status

from syrag import IngestRequest, IngestResponse


jobs: dict[str, IngestJob] = {}


async def run_ingest_job(job_id: str, request: IngestRequest) -> None:
    job = jobs[job_id]
    job.status = "running"
    try:
        job.result = await syrag.pipeline.run_ingest(
            request=request,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )
        job.status = "completed"
    except Exception as exc:
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
```

The enqueue route accepts the same `IngestRequest` model, normalizes defaults, schedules work, and returns `202 Accepted`:

```python
@syrag.post("/ingest-jobs", tags=["ingest"])
async def enqueue_ingest_job(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    response: Response,
) -> dict[str, str]:
    normalized_request = request.model_copy(
        update={
            "collection": request.collection or "support",
            "metadata": {"source": "background-job", **request.metadata},
        }
    )
    job_id = f"ingest-{len(jobs) + 1}"
    jobs[job_id] = IngestJob(job_id=job_id)
    background_tasks.add_task(run_ingest_job, job_id, normalized_request)
    response.status_code = status.HTTP_202_ACCEPTED
    return {
        "job_id": job_id,
        "status": jobs[job_id].status,
        "status_url": f"/ingest-jobs/{job_id}",
    }
```

The query route can stay a normal SyRAG route:

```python
@syrag.query("/query", embedder=embedder, vector_store=vector_store, llm=llm)
async def query(request: QueryRequest) -> QueryRequest:
    return request.model_copy(
        update={
            "collection": request.collection or "support",
            "top_k": min(request.top_k, 5),
        }
    )
```

## Call The Job API

Enqueue work:

```bash
curl -X POST http://127.0.0.1:8000/ingest-jobs \
  -H "content-type: application/json" \
  -d '{
    "documents": [
      "SyRAG can run document ingestion behind a job boundary.",
      "Query routes can continue serving while ingestion runs in the background."
    ],
    "collection": "support",
    "metadata": {"source_id": "background-guide"}
  }'
```

Example response:

```json
{
  "job_id": "ingest-1",
  "status": "queued",
  "status_url": "/ingest-jobs/ingest-1"
}
```

Poll status:

```bash
curl http://127.0.0.1:8000/ingest-jobs/ingest-1
```

Query after completion:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "content-type: application/json" \
  -d '{
    "query": "What can SyRAG do with ingestion?",
    "collection": "support",
    "top_k": 3
  }'
```

## Production Notes

- FastAPI `BackgroundTasks` are not durable. If the process restarts, queued work is lost.
- Store job state in Redis, Postgres, or another shared store when running multiple workers.
- Use idempotency keys or deterministic document IDs so retries do not duplicate chunks.
- Apply the same auth, tenant binding, safety limits, and observability used by synchronous routes.
- Keep large source files in object storage and pass references through the job payload instead of sending huge request bodies.
