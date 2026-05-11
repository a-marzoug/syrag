import os
import secrets
from pathlib import Path

from starlette.requests import Request

from syrag import (
    ChromaVectorStore,
    IngestRequest,
    OpenAIEmbedder,
    OpenAILLM,
    QueryRequest,
    RequestContext,
    Settings,
    SyRAG,
    SyRAGError,
)

SUPPORT_COLLECTION = "support"


class APIKeyAuthHook:
    """Authenticate requests with an x-api-key header and bind tenant context."""

    def __init__(self, *, api_keys: dict[str, str], header_name: str = "x-api-key") -> None:
        if not api_keys:
            msg = "api_keys must contain at least one tenant-to-key mapping"
            raise ValueError(msg)
        self.api_keys = api_keys
        self.header_name = header_name

    async def authenticate(
        self,
        *,
        request: Request,
        context: RequestContext,
    ) -> RequestContext:
        received_key = request.headers.get(self.header_name)
        if received_key is None:
            self._raise_auth_error(reason="missing_api_key")

        for tenant_id, expected_key in self.api_keys.items():
            if secrets.compare_digest(received_key, expected_key):
                return context.model_copy(
                    update={
                        "tenant_id": context.tenant_id or tenant_id,
                        "subject_id": f"api-key:{tenant_id}",
                        "auth_scheme": "api_key",
                        "scopes": ["rag:query", "rag:ingest"],
                        "metadata": {
                            **context.metadata,
                            "authenticated_tenant": tenant_id,
                        },
                    }
                )

        self._raise_auth_error(reason="invalid_api_key")

    def _raise_auth_error(self, *, reason: str) -> None:
        raise SyRAGError(
            code="authentication_failed",
            message="Failed to authenticate the request.",
            stage="auth",
            status_code=401,
            details={
                "component": type(self).__name__,
                "reason": reason,
            },
        )


def load_api_keys(raw_value: str) -> dict[str, str]:
    """Parse tenant:key pairs from SYRAG_API_KEYS."""

    api_keys: dict[str, str] = {}
    for item in raw_value.split(","):
        tenant_id, separator, api_key = item.partition(":")
        if separator and tenant_id.strip() and api_key.strip():
            api_keys[tenant_id.strip()] = api_key.strip()
    return api_keys


syrag = SyRAG(
    title="Authenticated Support Bot",
    version="0.1.0",
    description="SyRAG app protected by an API-key auth hook.",
    settings=Settings(),
)
syrag.set_auth_hook(
    APIKeyAuthHook(api_keys=load_api_keys(os.environ["SYRAG_API_KEYS"]))
)

embedder = OpenAIEmbedder(
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-3-small",
)
vector_store = ChromaVectorStore(
    path=Path(".syrag/auth-chroma"),
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
                "source": "authenticated-api",
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
