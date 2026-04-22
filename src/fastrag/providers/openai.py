from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from fastrag.protocols import LLM, Embedder
from fastrag.schemas import Citation, GenerationRequest, RAGResponse, RetrievedChunk


class OpenAIEmbedder(Embedder):
    """OpenAI-backed embedder using the embeddings API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        dimensions: int | None = None,
        timeout_seconds: float = 30.0,
        organization: str | None = None,
        project: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds
        self.organization = organization
        self.project = project
        self.transport = transport

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        payload: dict[str, Any] = {
            "model": self.model,
            "input": list(texts),
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        response_json = await self._post_json(path="/embeddings", payload=payload)
        raw_embeddings = sorted(
            response_json.get("data", []),
            key=lambda item: int(item.get("index", 0)),
        )
        if len(raw_embeddings) != len(texts):
            msg = "OpenAI embeddings response did not return one vector per input text"
            raise RuntimeError(msg)

        return [
            [float(value) for value in embedding["embedding"]]
            for embedding in raw_embeddings
        ]

    async def _post_json(
        self,
        *,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = await client.post(path, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPError as exc:
                msg = f"OpenAI embeddings request failed with status {response.status_code}"
                raise RuntimeError(msg) from exc
            return dict(response.json())

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization is not None:
            headers["OpenAI-Organization"] = self.organization
        if self.project is not None:
            headers["OpenAI-Project"] = self.project
        return headers


class OpenAILLM(LLM):
    """OpenAI-backed generator using the responses API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        organization: str | None = None,
        project: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.organization = organization
        self.project = project
        self.transport = transport

    async def generate(
        self,
        *,
        generation: GenerationRequest,
    ) -> RAGResponse:
        response_json = await self._post_json(
            path="/responses",
            payload={
                "model": self.model,
                "input": self._build_input(generation),
            },
        )
        answer = self._extract_text(response_json)
        return RAGResponse(
            answer=answer,
            citations=self._citations_for(generation.context, generation.require_citations),
            usage=self._usage_for(response_json),
        )

    def _build_input(self, generation: GenerationRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if generation.system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": generation.system_prompt,
                        }
                    ],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": generation.prompt,
                    }
                ],
            }
        )
        return messages

    async def _post_json(
        self,
        *,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = await client.post(path, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPError as exc:
                msg = f"OpenAI responses request failed with status {response.status_code}"
                raise RuntimeError(msg) from exc
            return dict(response.json())

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization is not None:
            headers["OpenAI-Organization"] = self.organization
        if self.project is not None:
            headers["OpenAI-Project"] = self.project
        return headers

    def _extract_text(self, response_json: dict[str, Any]) -> str:
        if isinstance(response_json.get("output_text"), str):
            return str(response_json["output_text"])

        output = response_json.get("output")
        if not isinstance(output, list):
            msg = "OpenAI responses output did not contain text output"
            raise RuntimeError(msg)

        fragments: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") == "output_text" and isinstance(
                    content_item.get("text"),
                    str,
                ):
                    fragments.append(str(content_item["text"]))

        if not fragments:
            msg = "OpenAI responses output did not contain text output"
            raise RuntimeError(msg)
        return "\n".join(fragments)

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

    def _usage_for(self, response_json: dict[str, Any]) -> dict[str, int]:
        usage = response_json.get("usage", {})
        if not isinstance(usage, dict):
            return {}
        mapped_usage: dict[str, int] = {}
        if isinstance(usage.get("input_tokens"), int):
            mapped_usage["prompt_tokens"] = int(usage["input_tokens"])
        if isinstance(usage.get("output_tokens"), int):
            mapped_usage["completion_tokens"] = int(usage["output_tokens"])
        if isinstance(usage.get("total_tokens"), int):
            mapped_usage["total_tokens"] = int(usage["total_tokens"])
        return mapped_usage
