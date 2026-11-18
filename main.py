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
INJ_SENT = re.compile(
    r"[^.!?\n]*(ignore (all )?(previous|prior)|disregard|system prompt|reveal[^.!?\n]*(prompt|secret)"
    r"|you are now|instead (say|respond|output|answer))[^.!?\n]*[.!?]?", re.I
)
COMPARE_RE = re.compile(r"\b(compare|versus|vs\.?|difference|differ|both|contrast|between)\b", re.I)
NUM_CUE = re.compile(r"\b(how many|how long|what (is|are) the|value|threshold|timeout|interval|port|size|quorum|limit|bits?)\b", re.I)

def stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t

def tokens(text: str) -> list[str]:
    return [stem(t) for t in (x.lower() for x in TOKEN_RE.findall(text)) if t not in STOP]

def clean_question(question: str) -> str:
    return INJ_SENT.sub(" ", question).strip()

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
    qlower = question.lower()
    phrases = [p for p in re.findall(r"[a-z0-9]+(?:\s+[a-z0-9]+){1,2}", qlower) if len(p) > 5]
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
        for phrase in phrases:
            if phrase in PASSAGES[i].lower():
                score += 1.2
        scores.append((i, score))
    return sorted(scores, key=lambda x: (-x[1], x[0]))

def best_sentences(question: str, passage: str, k: int = 2) -> str:
    qwords = set(tokens(question))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if s.strip()]
    want_num = bool(NUM_CUE.search(question))
    scored = []
    for idx, sentence in enumerate(sentences):
        score = len(qwords & set(tokens(sentence)))
        if want_num and re.search(r"\d", sentence):
            score += 0.5
        scored.append((score, idx, sentence))
    chosen = sorted(scored, key=lambda row: (-row[0], row[1]))[:k]
    chosen = [row for row in chosen if row[0] > 0] or scored[:1]
    return " ".join(sentence for _, _, sentence in sorted(chosen, key=lambda row: row[1]))

def pick_passages(question: str, ranked: list[tuple[int, float]]) -> list[int]:
    """Greedily select high-ranking passages that cover rare query terms."""
    if not ranked:
        return []
    rare = {term for term in set(tokens(question)) if 0 < DOC_FREQ.get(term, 0) <= 3}
    top_score = ranked[0][1]
    chosen: list[int] = []
    covered: set[str] = set()
    for index, score in ranked[:6]:
        new_terms = (rare & set(DOC_TOKENS[index])) - covered
        if not chosen or (new_terms and score >= 0.4 * top_score):
            chosen.append(index)
            covered |= new_terms
    return chosen[:3]

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
    print("Q:", req.question, "| top_k:", req.top_k, flush=True)
    question = clean_question(req.question)
    if not tokens(question):
        return Result(answer=None, sources=[], scores=[], confidence=0.05)
    ranked = rank(question)
    if not ranked:
        return Result(answer=None, sources=[], scores=[], confidence=0.05)
    best = ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    strength = 1.0 - math.exp(-max(0.0, best) / 3.0)
    margin = max(0.0, min(1.0, (best - second) / max(best, 0.35)))
    confidence = 0.65 * strength + 0.35 * margin
    # Require both a minimum evidence score and calibrated confidence to avoid guessing.
    if best < 0.75 or confidence < 0.34:
        return Result(answer=None, sources=[], scores=[], confidence=min(0.39, confidence))
    used = pick_passages(question, ranked)
    score_of = dict(ranked)
    pool = list(dict.fromkeys(used + [i for i, _ in ranked[:req.top_k]]))
    pool.sort(key=lambda i: (-score_of[i], i))
    pool = pool[:max(req.top_k, len(used))]
    top_score = max(score_of[pool[0]], 1e-9)
    normalized = [round(min(1.0, score_of[i] / top_score), 4) for i in pool]
    answer = " ".join(PASSAGES[i] for i in used)
    print("A:", answer, "| src:", pool, flush=True)
    return Result(answer=answer, sources=pool, scores=normalized,
                  confidence=round(max(0.61, min(0.99, confidence)), 4))

@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "passages": len(PASSAGES)}
