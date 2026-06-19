from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    answer_found: bool
    sources: list
    similarity_score: float = 0.0
    distance: float = 0.0


class AnalysticsResponse(BaseModel):
    most_frequent_question: list
    no_answer_questions: list
    avg_latency_ms: float
    p95_latency_ms: float
