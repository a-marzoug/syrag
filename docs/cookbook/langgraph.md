# LangGraph RAG Pipeline

This recipe uses LangGraph as the orchestration layer and SyRAG as the RAG service boundary. LangGraph owns the workflow. SyRAG owns retrieval, generation, citations, tenant context, guardrails, and observability.

The complete script is available at [`examples/integrations/langgraph_syrag_rag.py`](https://github.com/a-marzoug/syrag/blob/main/examples/integrations/langgraph_syrag_rag.py).

## Install

```bash
pip install langgraph langchain httpx "syrag[server]"
```

Run any SyRAG app with `/ingest` and `/query` routes first.

## Full Workflow

Create `langgraph_syrag_rag.py`:

```python
from __future__ import annotations

import asyncio
from typing import Any, TypedDict

import httpx
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph


class RAGState(TypedDict, total=False):
    question: str
    collection: str
    tenant_id: str
    answer: str
    citations: list[dict[str, Any]]


@tool
async def ask_syrag(
    question: str,
    collection: str = "support",
    tenant_id: str | None = None,
) -> str:
    """Ask the SyRAG knowledge service for a grounded answer with citations."""
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        response = await client.post(
            "/query",
            json={
                "query": question,
                "collection": collection,
                "tenant_id": tenant_id,
                "top_k": 5,
            },
            headers={
                "x-tenant-id": tenant_id,
            }
            if tenant_id
            else None,
        )
        response.raise_for_status()
        payload = response.json()
        citations = payload.get("citations", [])
        return f"{payload['answer']}\n\nCitations: {citations}"


agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[ask_syrag],
    system_prompt=(
        "You are a support agent. Use the ask_syrag tool for questions about "
        "the private knowledge base. Do not answer private-knowledge questions "
        "from memory."
    ),
)


async def run_agent(state: RAGState) -> RAGState:
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Question: {state['question']}\n"
                        f"Collection: {state.get('collection', 'support')}\n"
                        f"Tenant: {state.get('tenant_id') or ''}"
                    ),
                }
            ]
        }
    )
    return {"answer": str(result["messages"][-1].content)}


def build_graph() -> Any:
    graph = StateGraph(RAGState)
    graph.add_node("agent", run_agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


async def main() -> None:
    graph = build_graph()
    result = await graph.ainvoke(
        {
            "question": "What does SyRAG provide?",
            "collection": "support",
            "tenant_id": "tenant-a",
        }
    )
    print(result["answer"])
    print(result["citations"])


if __name__ == "__main__":
    asyncio.run(main())
```

Run:

```bash
python langgraph_syrag_rag.py
```

## Where To Add Real Logic

- Add a query-rewrite node before the agent when retrieval quality needs it.
- Add a moderation or routing node before the agent.
- Add a post-processing node after `agent` to format citations for Slack, web, or agents.
- Add LangGraph checkpointing when conversations need durable workflow state.
