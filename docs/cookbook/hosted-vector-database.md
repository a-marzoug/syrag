# Hosted Vector Database

Hosted vector databases are the preferred deployment shape once a SyRAG service needs persistence, backups, scaling, or independent operations ownership. This recipe uses hosted Qdrant because it has a simple Python client and the same payload filtering model as local Qdrant.

The complete hosted Qdrant example is available at [`examples/integrations/hosted_qdrant_app.py`](../../examples/integrations/hosted_qdrant_app.py).

## Hosted Qdrant App

Install:

```bash
pip install "syrag[openai,server]" qdrant-client
```

Configure the service:

```bash
export OPENAI_API_KEY="sk-..."
export QDRANT_URL="https://your-cluster.example-region.qdrant.io"
export QDRANT_API_KEY="qdrant-api-key"
export QDRANT_COLLECTION="support_docs"
export SYRAG_ENVIRONMENT="production"
```

Run:

```bash
uvicorn examples.integrations.hosted_qdrant_app:api --host 0.0.0.0 --port 8000
```

The app builds a remote `QdrantClient` from environment configuration:

```python
from qdrant_client import QdrantClient


client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ.get("QDRANT_API_KEY"),
    timeout=30,
)
```

The vector-store adapter creates the collection if it does not exist:

```python
if not self.client.collection_exists(collection_name):
    self.client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE,
        ),
    )
```

It stores SyRAG namespace fields in the payload so retrieval remains collection- and tenant-safe:

```python
payload={
    "chunk_id": chunk.chunk_id,
    "source_id": chunk.source_id,
    "content": chunk.content,
    "collection": collection or "",
    "tenant_id": tenant_id or "",
    "page_number": chunk.page_number,
    "chunk_index": chunk.chunk_index,
    "metadata": dict(chunk.metadata),
}
```

Query filters should always include the SyRAG collection and tenant fields before applying user metadata filters:

```python
models.Filter(
    must=[
        models.FieldCondition(
            key="collection",
            match=models.MatchValue(value=collection or ""),
        ),
        models.FieldCondition(
            key="tenant_id",
            match=models.MatchValue(value=tenant_id or ""),
        ),
    ]
)
```

## Call The Hosted App

Ingest:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "content-type: application/json" \
  -d '{
    "documents": ["Hosted Qdrant stores SyRAG chunks and vector payloads."],
    "collection": "support",
    "tenant_id": "tenant-a",
    "metadata": {"topic": "vector-database"}
  }'
```

Query:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "content-type: application/json" \
  -d '{
    "query": "What does hosted Qdrant store?",
    "collection": "support",
    "tenant_id": "tenant-a",
    "filters": {"topic": "vector-database"},
    "top_k": 3
  }'
```

## Deployment Notes

- Keep `QDRANT_URL`, `QDRANT_API_KEY`, and `OPENAI_API_KEY` in your platform secret store.
- Create collections during deployment if you do not want app startup to mutate infrastructure.
- Match `OPENAI_EMBEDDING_DIMENSIONS` to the embedding model and dimensions used by the app.
- Keep collection and tenant fields in payload filters even if your hosted database also has network-level isolation.
- Use separate Qdrant collections or clusters for materially different embedding dimensions.
- Add authentication and observability from the other cookbook recipes before exposing the service publicly.
