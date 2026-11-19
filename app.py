import json
import os
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

KB_PATH = Path(os.getenv("KB_PATH", "knowledge_base.json"))

# Lightweight ONNX embedding model; avoids PyTorch memory usage.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Load and validate the 50 passages.
with KB_PATH.open("r", encoding="utf-8") as f:
    raw_kb = json.load(f)

if raw_kb and isinstance(raw_kb[0], dict):
    passages = [str(item.get("text", "")).strip() for item in raw_kb]
else:
    passages = [str(item).strip() for item in raw_kb]

if len(passages) != 50:
    raise RuntimeError(f"Expected 50 passages, found {len(passages)}")

# Load embedding model and encode all passages once at startup.
embedding_model = TextEmbedding(model_name=MODEL_NAME)

passage_embeddings = np.array(
    list(embedding_model.embed(passages)),
    dtype=np.float32,
)

# Normalize rows so cosine similarity equals a dot product.
passage_embeddings /= np.linalg.norm(
    passage_embeddings,
    axis=1,
    keepdims=True,
)

app = FastAPI(title="RAG Colosseum")

# Required CORS configuration.
app.add_middleware(
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

COMPARISON_WORDS = {
    "compare", "difference", "differences", "versus", "vs",
    "between", "unlike", "similar", "similarities",
}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def keywords(text: str) -> set[str]:
    return {
        word
        for word in normalize(text).split()
        if word and word not in STOPWORDS
    }


def extract_answer(question: str, passages: list[str]) -> str | None:
    """Extract the most relevant factual sentence from retrieved passages."""
    qkw = keywords(question)
    candidates = []

    for passage in passages:
        sentences = re.split(r"(?<=[.!?])\s+", passage)

        for sentence in sentences:
            sentence_keywords = keywords(sentence)
            overlap = len(qkw & sentence_keywords)

            if overlap == 0:
                continue

            score = overlap / max(1, len(qkw))

            # Slightly prefer sentences that state a fact.
            if re.search(
                r"\b(is|are|requires|uses|use|provides|supports|listen|port|"
                r"default|means|ensures|prevents|reduces)\b",
                sentence,
                re.I,
            ):
                score += 0.15

            candidates.append(
                (score, len(sentence), sentence.strip())
            )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    answer = candidates[0][2]

    if len(answer) > 320:
        answer = answer[:320].rsplit(" ", 1)[0] + "…"

    return answer


def is_comparison(question: str) -> bool:
    return bool(keywords(question) & COMPARISON_WORDS)


def compare_answers(question: str, passages: list[str]) -> str | None:
    """Combine evidence from the top two passages for comparison questions."""
    if len(passages) < 2:
        return None

    facts = []

    for passage in passages[:2]:
        fact = extract_answer(question, [passage])
        if fact:
            facts.append(fact)

    if len(facts) < 2:
        return None

    return " | ".join(facts)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()

    # Abstain on empty questions and obvious prompt injections.
    if not question or any(
        re.search(pattern, question, re.I)
        for pattern in INJECTION_PATTERNS
    ):
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.0,
        )

    # Embed the question.
    q_embedding = np.array(
        list(embedding_model.embed([question])),
        dtype=np.float32,
    )[0]

    q_embedding /= np.linalg.norm(q_embedding)

    # Cosine similarity with every passage.
    similarities = passage_embeddings @ q_embedding
    ranked_indices = np.argsort(similarities)[::-1]

    top_k = min(request.top_k, 5)
    top_ids = [int(index) for index in ranked_indices[:top_k]]
    top_scores = [float(similarities[index]) for index in top_ids]

    # Abstain if the closest passage is not semantically relevant.
    if not top_scores or top_scores[0] < 0.35:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.0,
        )

    useful_passages = [
        passages[index]
        for index, score in zip(top_ids, top_scores)
        if score >= 0.35
    ]

    # Use two passages for comparison-style questions.
    if is_comparison(question):
        answer = compare_answers(question, useful_passages)
    else:
        answer = extract_answer(question, useful_passages)

    # Abstain if no supporting sentence can be extracted.
    if answer is None:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.0,
        )

    # Estimate confidence from retrieval quality and keyword evidence.
    qkw = keywords(question)
    evidence_overlap = 0.0

    for passage in useful_passages:
        overlap = len(qkw & keywords(passage)) / max(1, len(qkw))
        evidence_overlap = max(evidence_overlap, overlap)

    confidence = (0.55 * top_scores[0]) + (0.45 * evidence_overlap)
    confidence = max(0.0, min(1.0, confidence))

    # Enforce the abstention contract.
    if confidence < 0.40:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=confidence,
        )

    return QueryResponse(
        answer=answer,
        sources=top_ids,
        scores=[round(score, 4) for score in top_scores],
        confidence=round(confidence, 4),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
    )
