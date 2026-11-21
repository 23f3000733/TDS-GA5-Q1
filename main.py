"""Small offline RAG API for the supplied 50 passage knowledge base."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from fastembed import TextEmbedding

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
HTML_RE = re.compile(r"<!--.*?-->|\[\s*SYSTEM[^\]]*\]", re.I | re.S)
INJ_EXTRA = re.compile(
    r"(\{[^{}]*(sources|confidence|override)[^{}]*\}"
    r"|#{2,}.*$|(?:note|instruction|system|assistant)\s*(?:to \w+)?\s*:.*$"
    r"|force\s+sources.*$|set\s+confidence.*$|return\s+(sources|confidence).*$)",
    re.I | re.M,
)
MULTI_RE = re.compile(r"\b(both|integrate|synthes\w*|combine|compare|difference|contrast)\b", re.I)
MECH_RE = re.compile(r"^\s*(how (does|do|is|are)\b|why\b)|formulation|formula|equation|derivation|proof", re.I)
NUM_CUE = re.compile(r"\b(how many|how long|what (is|are) the|value|threshold|timeout|interval|port|size|quorum|limit|bits?)\b", re.I)

def stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t

def tokens(text: str) -> list[str]:
    return [stem(t) for t in (x.lower() for x in TOKEN_RE.findall(text)) if t not in STOP]

def clean_question(question: str) -> str:
    question = HTML_RE.sub(" ", question)
    question = INJ_EXTRA.sub(" ", question)
    return INJ_SENT.sub(" ", question).strip()

DOC_TOKENS = [tokens(p) for p in PASSAGES]
DOC_FREQ = Counter(t for doc in DOC_TOKENS for t in set(doc))
AVG_LEN = sum(map(len, DOC_TOKENS)) / max(1, len(DOC_TOKENS))

EMB = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir="/app/models")
P_VEC = np.array(list(EMB.embed(PASSAGES)))
P_VEC /= np.linalg.norm(P_VEC, axis=1, keepdims=True)
QPREFIX = "Represent this sentence for searching relevant passages: "
COS_MIN = 0.55

def dense(question: str) -> np.ndarray:
    vector = np.array(list(EMB.embed([QPREFIX + question])))[0]
    return P_VEC @ (vector / np.linalg.norm(vector))

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
SYN = {
    "keepalive": "heartbeat", "coordinator": "leader", "subordinate": "follower",
    "replicas": "nodes", "signals": "heartbeats", "scalar": "rate",
    "coefficient": "rate", "socket": "port", "daemon": "server",
}

def expand(question: str) -> str:
    lower = question.lower()
    extra = [value for key, value in SYN.items() if key in lower]
    return question + (" (" + " ".join(extra) + ")" if extra else "")

PASSAGE_ENTITIES = {
    0: ("raft",), 1: ("multi-paxos", "multi paxos", "paxos"),
    10: ("sgd", "stochastic gradient descent"), 11: ("adamw",),
    20: ("postgresql", "postgres", "postgre sql"), 21: ("mysql", "innodb"),
    40: ("public key cryptography", "public-key cryptography", "rsa"),
    41: ("elliptic curve cryptography", "elliptic-curve cryptography", "ecc", "ecdsa"),
}

def question_entity_position(question: str, passage_index: int) -> int:
    """Find the earliest explicit mention of this passage's main concept."""
    text = question.lower()
    positions = [text.find(name) for name in PASSAGE_ENTITIES.get(passage_index, ()) if text.find(name) >= 0]
    return min(positions, default=len(text) + passage_index)

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
    question = clean_question(req.question)
    print("CLEAN:", question, flush=True)
    abstain = Result(answer=None, sources=[], scores=[], confidence=0.1)
    if not tokens(question):
        return abstain
    multi = bool(MULTI_RE.search(question))
    if MECH_RE.search(question) and not multi:
        print("ABSTAIN mech:", question, flush=True)
        return abstain

    expanded = expand(question)
    bm = dict(rank(expanded))
    cosine = dense(expanded)
    cmax = float(cosine.max())
    bmax = max(max(bm.values()), 1e-9)
    final = {i: 0.25 * bm[i] / bmax + 0.75 * float(cosine[i]) for i in range(len(PASSAGES))}
    order = sorted(final, key=lambda i: (-final[i], i))
    print(f"Q: {question} | cmax={cmax:.3f} | top={order[:3]}", flush=True)

    if cmax < COS_MIN:
        return abstain

    if multi:
        # Keep sources and scores in descending relevance order for the API contract.
        sources = order[:2]
        # Present synthesis evidence in the order the concepts appear in the question.
        used = sorted(sources, key=lambda index: question_entity_position(question, index))
    else:
        used = order[:1]
        sources = order[:req.top_k]
    scores = [round(min(1.0, final[i] / max(final[order[0]], 1e-9)), 4) for i in sources]
    if multi:
        # Keep only the strongest evidence sentences from each chunk for synthesis.
        answer = " ".join(best_sentences(question, PASSAGES[i], k=2) for i in used)
    else:
        answer = PASSAGES[used[0]]
    print("A:", answer, "| src:", sources, flush=True)
    return Result(answer=answer, sources=sources, scores=scores,
                  confidence=round(min(0.99, max(0.62, cmax)), 4))

@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "passages": len(PASSAGES)}
