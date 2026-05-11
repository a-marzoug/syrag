# Authentication

SyRAG authentication belongs in an `AuthHook`, not inside each ingest or query handler. That keeps request identity, tenant binding, structured auth failures, and observability context consistent across every route.

The complete API-key example is available at [`examples/integrations/api_key_auth_app.py`](../../examples/integrations/api_key_auth_app.py).

## API-Key Protected SyRAG App

Use this pattern for internal services, prototypes, or deployments where an API gateway has already issued per-tenant service keys.

Install:

```bash
pip install "syrag[chroma,openai,server]"
```

Configure secrets:

```bash
export OPENAI_API_KEY="sk-..."
export SYRAG_API_KEYS="tenant-a:dev-key-a,tenant-b:dev-key-b"
```

Run the app:

```bash
uvicorn examples.integrations.api_key_auth_app:api --reload
```

The auth hook validates the `x-api-key` header and enriches SyRAG request context:

```python
import secrets

from starlette.requests import Request

from syrag import RequestContext, SyRAGError


class APIKeyAuthHook:
    def __init__(self, *, api_keys: dict[str, str], header_name: str = "x-api-key") -> None:
        self.api_keys = api_keys
        self.header_name = header_name

    async def authenticate(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> RequestContext:
        received_key = request.headers.get(self.header_name)
        for tenant_id, expected_key in self.api_keys.items():
            if received_key and secrets.compare_digest(received_key, expected_key):
                return context.model_copy(
                    update={
                        "tenant_id": context.tenant_id or tenant_id,
                        "subject_id": f"api-key:{tenant_id}",
                        "auth_scheme": "api_key",
                        "scopes": ["rag:query", "rag:ingest"],
                    }
                )

        raise SyRAGError(
            code="authentication_failed",
            message="Failed to authenticate the request.",
            stage="auth",
            status_code=401,
            details={"component": type(self).__name__},
        )
```

Register the hook once:

```python
syrag.set_auth_hook(APIKeyAuthHook(api_keys=load_api_keys(os.environ["SYRAG_API_KEYS"])))
```

Then keep route handlers focused on request normalization:

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

## Call The Protected Routes

Ingest:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "content-type: application/json" \
  -H "x-api-key: dev-key-a" \
  -d '{
    "documents": ["SyRAG protects routes with request-scope auth hooks."],
    "collection": "support"
  }'
```

Query:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "content-type: application/json" \
  -H "x-api-key: dev-key-a" \
  -d '{
    "query": "How does SyRAG protect routes?",
    "collection": "support",
    "top_k": 3
  }'
```

Invalid or missing keys return a structured `401` response:

```json
{
  "error": {
    "code": "authentication_failed",
    "message": "Failed to authenticate the request.",
    "stage": "auth",
    "details": {
      "component": "APIKeyAuthHook",
      "reason": "invalid_api_key"
    }
  }
}
```

## Production Notes

- Use HTTPS. API keys should never travel over plaintext HTTP.
- Store keys in a secret manager or platform secret store, not in source files.
- Rotate keys and support overlapping old/new keys during rollout.
- Prefer short-lived JWT/OIDC or gateway-authenticated identity for public internet APIs.
- Bind tenant identity in the auth hook so request payloads cannot self-select another tenant.
