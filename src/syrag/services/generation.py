from syrag.schemas import AssembledPrompt, GenerationRequest

DEFAULT_SYSTEM_PROMPT = (
    "Answer using the retrieved context when it is available. "
    "Be explicit when the available context is insufficient."
)


class DefaultGenerationPolicy:
    """Applies default generation constraints and instructions."""

    def __init__(
        self,
        *,
        max_context_chunks: int = 3,
        require_citations: bool = True,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if max_context_chunks <= 0:
            msg = "max_context_chunks must be a positive integer"
            raise ValueError(msg)
        self.max_context_chunks = max_context_chunks
        self.require_citations = require_citations
        self.system_prompt = system_prompt

    async def apply(
        self,
        *,
        prompt: AssembledPrompt,
    ) -> GenerationRequest:
        return GenerationRequest(
            query=prompt.query,
            context=list(prompt.context[: self.max_context_chunks]),
            prompt=prompt.prompt,
            system_prompt=self.system_prompt,
            require_citations=self.require_citations,
        )
