"""Small offline RAG API for the supplied 50 passage knowledge base."""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
with (HERE / "knowledge_base.json").open(encoding="utf-8") as f:
    PASSAGES: list[str] = json.load(f)

app = FastAPI(title="RAG Colosseum", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STOP = set("a an the and or but of in on at to for from by with as is are was were be been being do does did how what which who when where why can could should would about explain tell give list according".split())
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.I)
INJECTION_RE = re.compile(r"ignore (all )?(previous|prior) instructions|system prompt|reveal.*(prompt|secret)|disregard.*instruction", re.I)

def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOP]

DOC_TOKENS = [tokens(p) for p in PASSAGES]
DOC_FREQ = Counter(t for doc in DOC_TOKENS for t in set(doc))
AVG_LEN = sum(map(len, DOC_TOKENS)) / max(1, len(DOC_TOKENS))

# Small conceptual expansions help with common paraphrases while exact terms retain priority.
ALIASES = {
    "batch": ["minibatch", "samples"], "mini-batch": ["minibatch", "samples"],
    "timeout": ["timers", "interval", "duration"], "leader": ["raft", "consensus"],
    "password": ["credential", "secret", "hashing"], "hash": ["digest", "cryptographic"],
    "database": ["postgresql", "mysql", "mongodb", "cassandra", "redis"],
    "web": ["http", "browser", "tls"], "encryption": ["cipher", "aes", "chacha20"],
    "authentication": ["oauth", "jwt", "credential"], "rate": ["limiting", "bucket", "throttle"],
    "failure": ["dead", "suspicion", "eviction"], "causality": ["causal", "vector", "clock"],
}

def rank(question: str) -> list[tuple[int, float]]:
    q = tokens(question)
    if not q:
        return []
    qweights = Counter(q)
    for t in q:
        qweights.update({a: 0.45 for a in ALIASES.get(t, [])})
    scores = []
    n = len(PASSAGES)
    for i, doc in enumerate(DOC_TOKENS):
        counts = Counter(doc)
        score = 0.0
        for term, weight in qweights.items():
            tf = counts[term]
            if not tf:
                continue
            idf = math.log(1 + (n - DOC_FREQ[term] + 0.5) / (DOC_FREQ[term] + 0.5))
            norm = tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(doc) / AVG_LEN))
            score += weight * idf * norm
        # Phrase and named-entity overlap is valuable for distinguishing confounders.
        qlower = question.lower()
        for phrase in re.findall(r"[a-z0-9]+(?:\s+[a-z0-9]+){1,2}", qlower):
            if len(phrase) > 5 and phrase in PASSAGES[i].lower():
                score += 1.2
        scores.append((i, score))
    return sorted(scores, key=lambda x: (-x[1], x[0]))

def confidence_for(best: float, second: float, question: str) -> float:
    # BM25 score is transformed to [0,1]; score margin rejects weak/no-evidence matches.
    strength = 1.0 - math.exp(-max(0.0, best) / 3.0)
    margin = (best - second) / max(best, 0.35)
    conf = 0.65 * strength + 0.35 * max(0.0, min(1.0, margin))
    if INJECTION_RE.search(question):
        return 0.0
    return max(0.0, min(0.99, conf))

def answer_from(question: str, passages: list[str]) -> str:
    # Return factual sentence(s) from the evidence; never follow instructions embedded in KB text.
    qwords = set(tokens(question))
    sentences = [s.strip() for p in passages for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
    ranked = []
    for s in sentences:
        overlap = len(qwords & set(tokens(s)))
        ranked.append((overlap, s))
    ranked.sort(key=lambda x: (-x[0], len(x[1])))
    selected = [s for n, s in ranked[:2] if n > 0]
    return " ".join(selected) if selected else passages[0]

class Query(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=10)

class Result(BaseModel):
    answer: Optional[str]
    sources: list[int]
    scores: list[float]
    confidence: float

@app.post("/query", response_model=Result)
def query(req: Query) -> Result:
    if INJECTION_RE.search(req.question):
        return Result(answer=None, sources=[], scores=[], confidence=0.05)
    ranked = rank(req.question)
    if not ranked:
        return Result(answer=None, sources=[], scores=[], confidence=0.05)
    best, second = ranked[0][1], ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = confidence_for(best, second, req.question)
    # Require both a minimum evidence score and calibrated confidence to avoid guessing.
    if best < 0.75 or confidence < 0.34:
        return Result(answer=None, sources=[], scores=[], confidence=min(0.39, confidence))
    chosen = ranked[:req.top_k]
    raw = [s for _, s in chosen]
    top_score = max(raw[0], 1e-9)
    normalized = [round(min(1.0, max(0.0, s / top_score * (0.75 + 0.25 * (1 - math.exp(-s))))), 4) for s in raw]
    answer = answer_from(req.question, [PASSAGES[i] for i, _ in chosen[:2]])
    return Result(answer=answer, sources=[i for i, _ in chosen], scores=normalized, confidence=round(max(0.61, confidence), 4))

@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "passages": len(PASSAGES)}
