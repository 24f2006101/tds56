import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from fastapi import FastAPI
from fastembed import TextEmbedding
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"

with open(KB_PATH, "r", encoding="utf-8") as f:
    PASSAGES: List[str] = json.load(f)

app = FastAPI(title="RAG Colosseum Server")

_embed_model: TextEmbedding | None = None
_passage_embeddings: np.ndarray | None = None


def get_embedder() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        # Lightweight default model; instantiated only when first needed
        _embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embed_model


def get_passage_embeddings() -> np.ndarray:
    global _passage_embeddings
    if _passage_embeddings is None:
        model = get_embedder()
        vecs = list(model.embed(PASSAGES))
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        _passage_embeddings = arr / norms
    return _passage_embeddings


def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9\.\-/]+", text.lower()))


def retrieve_passages(query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
    model = get_embedder()
    p_emb = get_passage_embeddings()
    q_vec = np.array(list(model.embed([query]))[0], dtype=np.float32)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)

    dense_scores = p_emb @ q_vec
    q_tokens = tokenize(query)

    results = []
    for idx, passage in enumerate(PASSAGES):
        p_tokens = tokenize(passage)
        overlap = len(q_tokens & p_tokens) / max(len(q_tokens), 1)
        score = float(0.80 * dense_scores[idx] + 0.20 * overlap)
        results.append((idx, passage, score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "passages": len(PASSAGES)}


@app.post("/query")
def query_kb(req: QueryRequest):
    hits = retrieve_passages(req.question, top_k=5)
    return {
        "question": req.question,
        "matches": [
            {"id": idx, "score": round(score, 4), "passage": text}
            for idx, text, score in hits
        ],
    }
