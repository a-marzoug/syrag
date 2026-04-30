# LangGraph RAG Pipeline

This recipe uses LangGraph as the orchestration layer and SyRAG as the RAG service boundary. LangGraph owns the workflow. SyRAG owns retrieval, generation, citations, tenant context, guardrails, and observability.

The complete script is available at [`examples/integrations/langgraph_syrag_rag.py`](../../examples/integrations/langgraph_syrag_rag.py).

## Install

```bash
pip install langgraph langchain-core httpx "syrag[server]"
```

Run any SyRAG app with `/ingest` and `/query` routes first.

## Full Workflow

Create `langgraph_syrag_rag.py`:

```python
from __future__ import annotations

import asyncio
from typing import Any, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph


class RAGState(TypedDict, total=False):
    question: str
    collection: str
    tenant_id: str
    rewritten_query: str
    answer: str
    citations: list[dict[str, Any]]


async def rewrite_query(state: RAGState) -> RAGState:
    question = state["question"].strip()
    return {
        "rewritten_query": question,
    }


async def call_syrag(state: RAGState) -> RAGState:
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        response = await client.post(
            "/query",
            json={
                "query": state["rewritten_query"],
                "collection": state.get("collection", "support"),
                "tenant_id": state.get("tenant_id"),
                "top_k": 5,
            },
            headers={
                "x-tenant-id": state["tenant_id"],
            }
            if state.get("tenant_id")
            else None,
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "answer": payload["answer"],
            "citations": payload.get("citations", []),
        }


def build_graph() -> Any:
    graph = StateGraph(RAGState)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("call_syrag", call_syrag)
    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "call_syrag")
    graph.add_edge("call_syrag", END)
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

- Replace `rewrite_query` with an LLM call when query rewriting is valuable.
- Add a moderation or routing node before `call_syrag`.
- Add a post-processing node after `call_syrag` to format citations for Slack, web, or agents.
- Add LangGraph checkpointing when conversations need durable workflow state.
