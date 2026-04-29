from collections.abc import Sequence

from syrag.schemas import AssembledPrompt, QueryRequest, RetrievedChunk


class DefaultPromptAssembler:
    """Builds a deterministic grounded prompt from query inputs and retrieved chunks."""

    async def assemble(
        self,
        *,
        query: QueryRequest,
        context: Sequence[RetrievedChunk],
    ) -> AssembledPrompt:
        context_chunks = list(context)
        return AssembledPrompt(
            query=query,
            context=context_chunks,
            prompt=self._render_prompt(query=query, context=context_chunks),
        )

    def _render_prompt(
        self,
        *,
        query: QueryRequest,
        context: Sequence[RetrievedChunk],
    ) -> str:
        return "\n".join(
            [
                "You are a grounded RAG assistant.",
                f"Question: {query.query}",
                "Retrieved context:",
                self._render_context(context),
                "Answer using the retrieved context when it is available.",
            ]
        )

    def _render_context(self, context: Sequence[RetrievedChunk]) -> str:
        if not context:
            return "No retrieved context was available."

        rendered_chunks: list[str] = []
        for index, chunk in enumerate(context, start=1):
            page_suffix = f", page {chunk.page_number}" if chunk.page_number is not None else ""
            rendered_chunks.append(
                f"{index}. [{chunk.source_id}{page_suffix}] {chunk.content}"
            )
        return "\n".join(rendered_chunks)
