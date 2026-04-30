import httpx
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def ask_syrag(question: str, collection: str = "support", tenant_id: str | None = None) -> str:
    """Ask the SyRAG knowledge service for a grounded answer with citations."""
    with httpx.Client(base_url="http://127.0.0.1:8000") as client:
        response = client.post(
            "/query",
            json={
                "query": question,
                "collection": collection,
                "tenant_id": tenant_id,
                "top_k": 5,
            },
            headers={"x-tenant-id": tenant_id} if tenant_id else None,
        )
        response.raise_for_status()
        payload = response.json()
        return f"{payload['answer']}\n\nCitations: {payload.get('citations', [])}"


agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[ask_syrag],
    system_prompt=(
        "You are a support agent. Use ask_syrag for questions about the private "
        "knowledge base. Do not answer private-knowledge questions from memory."
    ),
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use the support collection for tenant-a. "
                        "What does SyRAG provide?"
                    ),
                }
            ]
        }
    )
    print(result["messages"][-1].content)
