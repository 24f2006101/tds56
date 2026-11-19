import json
import os
import re
from pathlib import Path

import gradio as gr
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

KB_PATH = Path(os.getenv("KB_PATH", "knowledge_base.json"))
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

with KB_PATH.open("r", encoding="utf-8") as f:
    raw_kb = json.load(f)

if raw_kb and isinstance(raw_kb[0], dict):
    passages = [str(item.get("text", "")).strip() for item in raw_kb]
else:
    passages = [str(item).strip() for item in raw_kb]

if len(passages) != 50:
    raise RuntimeError(f"Expected 50 passages, found {len(passages)}")

model = SentenceTransformer(MODEL_NAME)
passage_embeddings = model.encode(
    passages,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
)

api = FastAPI(title="RAG Colosseum")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str | None
    sources: list[int]
    scores: list[float]
    confidence: float

STOPWORDS = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "is", "are", "was", "were", "does", "do", "did", "the", "a", "an", "of",
    "in", "on", "for", "to", "and", "or", "with", "by", "from", "that", "this",
    "it", "its", "as", "at", "be", "can", "could", "should", "would", "please",
    "tell", "me", "about", "according", "based", "knowledge", "base", "passage",
}

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (all |any )?(previous|prior|above) instructions",
    r"you are now",
    r"system prompt",
    r"reveal your (instructions|prompt|system)",
    r"print your (instructions|prompt|system)",
    r"jailbreak",
    r"developer mode",
]

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()

def keywords(text: str):
    return {w for w in normalize(text).split() if w and w not in STOPWORDS}

def extract_answer(question: str, passages: list[str]) -> str | None:
    qkw = keywords(question)
    candidates = []

    for passage in passages:
        for sentence in re.split(r"(?<=[.!?])\s+", passage):
            overlap = len(qkw & keywords(sentence))
            if not overlap:
                continue

            score = overlap / max(1, len(qkw))
            if re.search(
                r"\b(is|are|requires|uses|use|provides|supports|listen|port|default|means|ensures|prevents|reduces)\b",
                sentence,
                re.I,
            ):
                score += 0.15

            candidates.append((score, len(sentence), sentence.strip()))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    answer = candidates[0][2]

    if len(answer) > 320:
        answer = answer[:320].rsplit(" ", 1)[0] + "…"

    return answer

def is_comparison(question: str) -> bool:
    return bool(keywords(question) & {
        "compare", "difference", "differences", "versus", "vs",
        "between", "unlike", "similar", "similarities",
    })

def compare_answers(question: str, passages: list[str]) -> str | None:
    if len(passages) < 2:
        return None

    facts = []
    for passage in passages[:2]:
        fact = extract_answer(question, [passage])
        if fact:
            facts.append(fact)

    return " | ".join(facts) if len(facts) == 2 else None

@api.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()

    if not question or any(
        re.search(pattern, question, re.I) for pattern in INJECTION_PATTERNS
    ):
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.0)

    q_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    similarities = passage_embeddings @ q_embedding
    ranked = np.argsort(similarities)[::-1]

    top_k = min(request.top_k, 5)
    top_ids = [int(i) for i in ranked[:top_k]]
    top_scores = [float(similarities[i]) for i in top_ids]

    if not top_scores or top_scores[0] < 0.35:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.0)

    useful = [
        passages[i]
        for i, score in zip(top_ids, top_scores)
        if score >= 0.35
    ]

    answer = compare_answers(question, useful) if is_comparison(question) else extract_answer(question, useful)

    if answer is None:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.0)

    qkw = keywords(question)
    evidence_overlap = max(
        (len(qkw & keywords(p)) / max(1, len(qkw)) for p in useful),
        default=0.0,
    )

    confidence = 0.55 * top_scores[0] + 0.45 * evidence_overlap
    confidence = max(0.0, min(1.0, confidence))

    if confidence < 0.40:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=confidence)

    return QueryResponse(
        answer=answer,
        sources=top_ids,
        scores=[round(s, 4) for s in top_scores],
        confidence=round(confidence, 4),
    )

# Required: expose the FastAPI API through Gradio's server.
app = gr.mount_gradio_app(api, gr.Blocks(), path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))