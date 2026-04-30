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
