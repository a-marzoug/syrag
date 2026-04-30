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
    return {"rewritten_query": state["question"].strip()}


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
