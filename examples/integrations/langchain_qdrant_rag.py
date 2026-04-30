import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION = "support_docs"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
)
client = QdrantClient(path=".syrag/langchain-qdrant")
vector_size = len(embeddings.embed_query("dimension probe"))

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION,
    embedding=embeddings,
)

documents = [
    Document(
        page_content="SyRAG exposes typed ingest and query routes for RAG services.",
        metadata={"source": "overview", "topic": "api"},
    ),
    Document(
        page_content="Qdrant stores vectors and payloads for similarity search.",
        metadata={"source": "qdrant", "topic": "vector-store"},
    ),
]
vector_store.add_documents(documents)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Answer from the provided context. If the context is insufficient, "
                "say you do not know."
            ),
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)
model = ChatOpenAI(model="gpt-4.1-mini", api_key=os.environ["OPENAI_API_KEY"])


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

if __name__ == "__main__":
    answer = rag_chain.invoke("What does SyRAG expose?")
    print(answer)
