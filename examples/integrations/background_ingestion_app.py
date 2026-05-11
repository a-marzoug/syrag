import os
from dataclasses import dataclass, field
from pathlib import Path
from time import time

from fastapi import BackgroundTasks, Response, status

from syrag import (
    ChromaVectorStore,
    IngestRequest,
    IngestResponse,
    OpenAIEmbedder,
    OpenAILLM,
    PassThroughChunker,
    QueryRequest,
    Settings,
    SyRAG,
)

SUPPORT_COLLECTION = "support"


@dataclass
class IngestJob:
    job_id: str
    status: str = "queued"
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)
    result: IngestResponse | None = None
    error: str | None = None


jobs: dict[str, IngestJob] = {}

syrag = SyRAG(
    title="Background Ingestion Support Bot",
    version="0.1.0",
    description="SyRAG app with asynchronous ingestion jobs.",
    settings=Settings(),
)

chunker = PassThroughChunker()
embedder = OpenAIEmbedder(
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-3-small",
)
vector_store = ChromaVectorStore(
    path=Path(".syrag/background-ingestion-chroma"),
    collection_name="support_docs",
)
llm = OpenAILLM(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4.1-mini",
)


async def run_ingest_job(job_id: str, request: IngestRequest) -> None:
    job = jobs[job_id]
    job.status = "running"
    job.updated_at = time()
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
    finally:
        job.updated_at = time()


@syrag.post("/ingest-jobs", tags=["ingest"])
async def enqueue_ingest_job(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    response: Response,
) -> dict[str, str]:
    normalized_request = request.model_copy(
        update={
            "collection": request.collection or SUPPORT_COLLECTION,
            "metadata": {
                "source": "background-job",
                **request.metadata,
            },
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


@syrag.get("/ingest-jobs/{job_id}", tags=["ingest"])
async def get_ingest_job(job_id: str) -> dict[str, object]:
    job = jobs[job_id]
    return {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "result": job.result.model_dump() if job.result is not None else None,
        "error": job.error,
    }


@syrag.query("/query", embedder=embedder, vector_store=vector_store, llm=llm)
async def query(request: QueryRequest) -> QueryRequest:
    return request.model_copy(
        update={
            "collection": request.collection or SUPPORT_COLLECTION,
            "top_k": min(request.top_k, 5),
        }
    )


api = syrag.api
