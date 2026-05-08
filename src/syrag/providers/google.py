from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from syrag._optional import missing_optional_dependency
from syrag.errors import ProviderRequestError, ProviderResponseError
from syrag.protocols import LLM, Embedder
from syrag.schemas import Citation, GenerationRequest, RAGResponse, RetrievedChunk

try:
    from google import genai
    from google.genai import types
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via import path
    raise missing_optional_dependency(
        feature="syrag.providers.google",
        extra="google",
    ) from exc


class GoogleEmbedder(Embedder):
    """Google Gen AI-backed embedder using Gemini embedding models."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
        output_dimensionality: int | None = None,
        task_type: str | None = None,
        vertexai: bool | None = None,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        self.model = model
        self.client = client or genai.Client(
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
        )
        self.output_dimensionality = output_dimensionality
        self.task_type = task_type

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            response = await self.client.aio.models.embed_content(
                model=self.model,
                contents=cast(Any, list(texts)),
                config=self._config(),
            )
        except Exception as exc:
            raise ProviderRequestError(
                code="provider_request_failed",
                message="Google embeddings request failed",
                details={"provider": "google", "operation": "embed_content"},
            ) from exc

        return self._extract_embeddings(response, expected_count=len(texts))

    def _config(self) -> types.EmbedContentConfigDict | None:
        config: types.EmbedContentConfigDict = {}
        if self.output_dimensionality is not None:
            config["output_dimensionality"] = self.output_dimensionality
        if self.task_type is not None:
            config["task_type"] = self.task_type
        if not config:
            return None
        return config

    def _extract_embeddings(self, response: Any, *, expected_count: int) -> list[list[float]]:
        raw_embeddings = getattr(response, "embeddings", None)
        if not isinstance(raw_embeddings, list) or len(raw_embeddings) != expected_count:
            raise ProviderResponseError(
                code="provider_invalid_response",
                message="Google embeddings response did not return one vector per input text",
                details={"provider": "google", "operation": "embed_content"},
            )

        embeddings: list[list[float]] = []
        for raw_embedding in raw_embeddings:
            values = getattr(raw_embedding, "values", None)
            if not isinstance(values, list) or not values:
                raise ProviderResponseError(
                    code="provider_invalid_response",
                    message="Google embeddings response contained an empty vector",
                    details={"provider": "google", "operation": "embed_content"},
                )
            embeddings.append([float(value) for value in values])
        return embeddings


class GoogleLLM(LLM):
    """Google Gen AI-backed generator using Gemini models."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
        vertexai: bool | None = None,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        self.model = model
        self.client = client or genai.Client(
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
        )

    async def generate(
        self,
        *,
        generation: GenerationRequest,
    ) -> RAGResponse:
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=generation.prompt,
                config=self._config_for(generation),
            )
        except Exception as exc:
            raise ProviderRequestError(
                code="provider_request_failed",
                message="Google generation request failed",
                details={"provider": "google", "operation": "generate_content"},
            ) from exc

        answer = self._extract_text(response)
        return RAGResponse(
            answer=answer,
            citations=self._citations_for(generation.context, generation.require_citations),
            usage=self._usage_for(response),
        )

    def _config_for(
        self,
        generation: GenerationRequest,
    ) -> types.GenerateContentConfigDict | None:
        if generation.system_prompt is None:
            return None
        return {"system_instruction": generation.system_prompt}

    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ProviderResponseError(
                code="provider_invalid_response",
                message="Google generation response did not contain text output",
                details={"provider": "google", "operation": "generate_content"},
            )
        return text

    def _citations_for(
        self,
        context: Sequence[RetrievedChunk],
        require_citations: bool,
    ) -> list[Citation]:
        if not require_citations:
            return []
        return [
            Citation(
                source_id=document.source_id,
                score=document.score,
                snippet=document.content,
                page_number=document.page_number,
            )
            for document in context
        ]

    def _usage_for(self, response: Any) -> dict[str, int]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {}
        mapped_usage: dict[str, int] = {}
        if isinstance(getattr(usage, "prompt_token_count", None), int):
            mapped_usage["prompt_tokens"] = int(usage.prompt_token_count)
        if isinstance(getattr(usage, "candidates_token_count", None), int):
            mapped_usage["completion_tokens"] = int(usage.candidates_token_count)
        if isinstance(getattr(usage, "total_token_count", None), int):
            mapped_usage["total_tokens"] = int(usage.total_token_count)
        return mapped_usage
