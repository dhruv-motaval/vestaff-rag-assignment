from app.ingest import embedding_model
from app.ingest import chroma_client

from fastapi import HTTPException, status

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from app.config import TOP_K, GROQ_API_KEY, SIMILARITY_THRESHOLD

llm = init_chat_model("groq:llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

prompt = ChatPromptTemplate.from_template(
    """
You are a Retrieval-Augmented Generation (RAG) assistant.

Answer questions using ONLY the retrieved context provided below

Rules:
- Do not invent facts.
- Do not use external knowledge.
- If the answer is not explicitly supported by the context, respond exactly:

Information not found in document.

Retrieved Context:
{context}

Question: 
{question}

Answer:
"""
)


def get_collection():
    """Returns the existing chromaDB collection, Assumes ingest has already been run."""

    return chroma_client.get_collection(name="document_collection")


def retrieve_documents(question: str):
    """Encodes the question using sentence-transformers and queries ChromaDB for the top-k most similar chunks by cosine distance."""

    collection = get_collection()

    query_embedding = embedding_model.encode(question).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)

    return results


def answer_question(question: str):
    """Full RAG pipeline for given question.
    - Verifies the chromaDB collection exists, raises 404 if not.
    - Retrieves top-k chunks via cosine similarity.
    - Returns 'not found' early if best distance exceeds SIMILARITY_THRESHOLD.
    - Constructs a prompt with retrieved context and calls the Groq LLM.
    - Returns the answer, answer_found flag, sources, similarity score, and distance.
    """

    try:
        get_collection()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No documents ingested yet. Please call /ingest First.",
        )

    results = retrieve_documents(question)

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return {
            "answer": "Information not found in document.",
            "answer_found": False,
            "sources": [],
        }

    best_distance = distances[0]

    if best_distance > SIMILARITY_THRESHOLD:
        return {
            "answer": "Information not found in document.",
            "answer_found": False,
            "sources": [],
            "similarity_score": 0.0,
            "distance": round(best_distance, 4),
        }

    context = "\n\n".join(documents)

    chain = prompt | llm

    response = chain.invoke({"context": context, "question": question})

    answer = response.content.strip()

    answer_found = "information not found in document" not in answer.lower()

    sources = []

    for doc, meta in zip(documents, metadatas):
        sources.append({"page": meta["page"], "snippet": doc[:250].strip() + "..."})

    return {
        "answer": answer,
        "answer_found": answer_found,
        "sources": sources,
        "similarity_score": round(max(0.0, 1 - best_distance), 4),
        "distance": round(best_distance, 4),
    }
