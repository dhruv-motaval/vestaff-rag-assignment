from app.schemas import AskResponse
from app.schemas import AnalysticsResponse
from fastapi import FastAPI, HTTPException, status

import time

from app.ingest import ingest_document
from app.rag import answer_question
from app.analytics import get_analytics
from app.db import init_db, log_query
from app.schemas import AskRequest

app = FastAPI()


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def root():

    return {"message": "RAG API RUNNING"}


@app.get("/health")
def health():

    return {"status": "healthy"}


@app.post("/ingest")
def ingest():
    print("INGEST ENDPOINT HIT")

    result = ingest_document()

    print(result)
    return result


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid question"
        )

    start_time = time.time()

    result = answer_question(question)

    latency_ms = (time.time() - start_time) * 1000

    log_query(
        question=question, answer_found=result["answer_found"], latency_ms=latency_ms
    )

    return result


@app.get("/analytics", response_model=AnalysticsResponse)
def analystics():
    return get_analytics()
