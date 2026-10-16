"""
Adaptive Hybrid RAG for Enterprise QA
MSc Dissertation — Final Implementation
========================================

Eight configurations (RQ1-RQ3):
  sparse_bm25               BM25 only,             no citations  (RQ1 baseline)
  dense                     Dense (FAISS) only,    no citations  (RQ1 baseline)
  static_hybrid             BM25+Dense RRF,        no citations  (RQ1 treatment)
  static_hybrid_cite        BM25+Dense RRF,        cited         (RQ2)
  static_hybrid_weighted    BM25+Dense weighted,   no citations  (RQ3 baseline)
  static_hybrid_weighted_cite BM25+Dense weighted, cited         (RQ3 baseline)
  adaptive_hybrid           Heuristic router,      no citations  (RQ3 treatment)
  adaptive_hybrid_cite      Heuristic router,      cited         (RQ3 treatment)

RQ3 design note: static_hybrid_weighted uses the same weighted_fusion function as
adaptive_hybrid but with a fixed dense_weight=0.5, so the ONLY difference between
the RQ3 baseline and treatment is whether the weight is fixed or query-adaptive.
This eliminates the fusion-method confound that would arise from comparing RRF vs
weighted fusion across the RQ3 pair.

Usage (after adding API keys to .env):
  python ahrag.py --datasets sample --no-ragas --limit 6
  python ahrag.py --datasets financebench cuad azure
  python ahrag.py --datasets sample --dry-run
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

# ragas.executor calls nest_asyncio.apply() at import time. nest_asyncio's
# monkey-patch breaks asyncio.current_task() tracking on Python >= 3.13, which
# makes every RAGAS metric fail with "Timeout should be used inside a task"
# before any judge call is made. The patch exists only for Jupyter-style nested
# event loops; this script runs with no outer loop, so it is never needed here.
try:
    import nest_asyncio
    nest_asyncio.apply = lambda *a, **k: None
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 1. DATA TYPES
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    title: str = ""
    position: int = 0

    @property
    def citation_label(self) -> str:
        return self.title or self.doc_id


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class QAItem:
    qid: str
    question: str
    reference_answer: Optional[str] = None
    gold_doc_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class GeneratedAnswer:
    text: str
    citations: list[str] = field(default_factory=list)
    citation_coverage: float = float("nan")
    invalid_citation_rate: float = float("nan")
    raw: str = ""


# ---------------------------------------------------------------------------
# 2. CHUNKING  (tiktoken-aware, falls back to whitespace)
# ---------------------------------------------------------------------------

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def _encode(text: str) -> list:
    return _ENC.encode(text) if _ENC else text.split()


def _decode(tokens: list) -> str:
    return _ENC.decode(tokens) if _ENC else " ".join(tokens)


def chunk_document(
    doc: Document, chunk_size: int = 512, chunk_overlap: int = 64
) -> list[Chunk]:
    """Split a document into overlapping token windows."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")
    tokens = _encode(doc.text)
    if not tokens:
        return []
    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    pos = 0
    start = 0
    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        text = _decode(window).strip()
        if text:
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}::{pos}",
                doc_id=doc.doc_id,
                text=text,
                title=doc.title,
                position=pos,
            ))
            pos += 1
        if start + chunk_size >= len(tokens):
            break
        start += step
    return chunks


def chunk_corpus(
    docs: list[Document], chunk_size: int = 512, chunk_overlap: int = 64
) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, chunk_size, chunk_overlap))
    return out


# ---------------------------------------------------------------------------
# 3. EMBEDDINGS
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray: ...


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class SentenceTransformerEmbedder:
    """Local retriever embeddings — all-MiniLM-L6-v2 (no API cost)."""

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model, device=device)
        self.model = model
        self.batch_size = batch_size
        _dim_fn = getattr(self._model, "get_embedding_dimension", None) or \
                  getattr(self._model, "get_sentence_embedding_dimension", None)
        self.dim = int(_dim_fn())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class OpenAIEmbedder:
    """OpenAI embeddings — used only for the RAGAS judge, not the retriever."""

    _DIMS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: str | None = None,
        batch_size: int = 128,
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        from openai import OpenAI
        self._client = OpenAI(api_key=key)
        self.model = model
        self.dim = self._DIMS.get(model, 3072)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + self.batch_size]]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            vecs.extend([d.embedding for d in resp.data])
        arr = np.asarray(vecs, dtype=np.float32)
        self.dim = arr.shape[1]
        return arr


def make_dense_embedder(
    model: str = "all-MiniLM-L6-v2",
    backend: str = "sentence-transformers",
) -> Embedder:
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model)
    if backend == "openai":
        return OpenAIEmbedder(model)
    raise ValueError(f"Unknown embedding backend: {backend!r}")


# ---------------------------------------------------------------------------
# 4. LLM CLIENTS
# ---------------------------------------------------------------------------

class ChatLLM(Protocol):
    model: str
    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str: ...


class OpenAIChat:
    """GPT-4.1 generator (fixed per dissertation proposal)."""

    def __init__(self, model: str = "gpt-4.1", api_key: str | None = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set.\n"
                'Set it before running: $env:OPENAI_API_KEY = "sk-..."'
            )
        from openai import OpenAI
        self._client = OpenAI(api_key=key)
        self.model = model

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        import random
        import time as _time
        from openai import (
            APIConnectionError, APITimeoutError, InternalServerError, RateLimitError,
        )
        # Tier-limited accounts (e.g. 30k TPM) hit 429s under sustained load; a
        # single unretried 429 would abort the whole study mid-config.
        delay = 5.0
        for attempt in range(8):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
            except (RateLimitError, APIConnectionError, APITimeoutError,
                    InternalServerError) as exc:
                if attempt == 7:
                    raise
                wait = delay + random.uniform(0.0, 2.0)
                hint = re.search(r"try again in ([\d.]+)s", str(exc))
                if hint:
                    wait = float(hint.group(1)) + random.uniform(0.5, 2.0)
                _time.sleep(wait)
                delay = min(delay * 2, 90.0)
        raise RuntimeError("unreachable")


class AnthropicChat:
    """Claude (optional secondary / judge)."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set.\n"
                'Set it before running: $env:ANTHROPIC_API_KEY = "sk-ant-..."'
            )
        import anthropic
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        ).strip()


def make_generator(model: str = "gpt-4.1") -> ChatLLM:
    return OpenAIChat(model=model)


def make_judge_llm(provider: str = "openai", model: str = "gpt-4o"):
    """LangChain-wrapped LLM for RAGAS."""
    from ragas.llms import LangchainLLMWrapper
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY required for RAGAS judge.")
        return LangchainLLMWrapper(ChatOpenAI(model=model, temperature=0.0))
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY required for RAGAS judge.")
        return LangchainLLMWrapper(ChatAnthropic(model=model, temperature=0.0))
    raise ValueError(f"Unknown judge provider: {provider!r}")


# ---------------------------------------------------------------------------
# 5. RETRIEVAL — BASE
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenisation shared by BM25 and the router."""
    return _TOKEN_RE.findall(text.lower())


class Retriever(ABC):
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[Retrieved]: ...

    def __len__(self) -> int:
        return len(self.chunks)


# ---------------------------------------------------------------------------
# 6. RETRIEVAL — SPARSE (BM25 Okapi)
# ---------------------------------------------------------------------------

class BM25Retriever(Retriever):
    def __init__(self, chunks: list[Chunk]):
        super().__init__(chunks)
        self._bm25 = None
        if not chunks:
            return
        from rank_bm25 import BM25Okapi
        corpus_tokens = [tokenize(c.text) for c in chunks]
        safe = [toks if toks else ["<empty>"] for toks in corpus_tokens]
        self._bm25 = BM25Okapi(safe)

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        if not self.chunks or self._bm25 is None:
            return []
        q = tokenize(query) or ["<empty>"]
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            Retrieved(chunk=self.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]


# ---------------------------------------------------------------------------
# 7. RETRIEVAL — DENSE (FAISS cosine via all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------

class DenseRetriever(Retriever):
    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        super().__init__(chunks)
        self.embedder = embedder
        self._index = None
        self._build()

    def _build(self) -> None:
        if not self.chunks:
            return
        import faiss
        mat = l2_normalize(
            self.embedder.embed([c.text for c in self.chunks]).astype(np.float32)
        )
        self._dim = mat.shape[1]
        index = faiss.IndexFlatIP(self._dim)   # inner product on L2-normalised = cosine
        index.add(mat)
        self._index = index

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        if not self.chunks or self._index is None:
            return []
        q = l2_normalize(self.embedder.embed([query]).astype(np.float32))
        k = min(k, len(self.chunks))
        scores, idx = self._index.search(q, k)
        out: list[Retrieved] = []
        for rank, (i, s) in enumerate(zip(idx[0], scores[0]), start=1):
            if i < 0:
                continue
            out.append(Retrieved(chunk=self.chunks[int(i)], score=float(s), rank=rank))
        return out


# ---------------------------------------------------------------------------
# 8. RETRIEVAL — FUSION UTILITIES
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    sparse: list[Retrieved],
    dense: list[Retrieved],
    k: int,
    rrf_k: int = 60,
) -> list[Retrieved]:
    """RRF: rank-robust combination that ignores incomparable raw score scales."""
    scores: dict[str, float] = {}
    chunk_of: dict[str, Retrieved] = {}
    for ranked in (sparse, dense):
        for r in ranked:
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + r.rank)
            chunk_of[cid] = r
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        Retrieved(chunk=chunk_of[cid].chunk, score=score, rank=rank)
        for rank, (cid, score) in enumerate(ordered, start=1)
    ]


def _min_max(values: list[float]) -> dict[int, float]:
    if not values:
        return {}
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return {i: 1.0 for i in range(len(values))}
    return {i: (v - lo) / span for i, v in enumerate(values)}


def weighted_fusion(
    sparse: list[Retrieved],
    dense: list[Retrieved],
    k: int,
    dense_weight: float = 0.5,
) -> list[Retrieved]:
    """Min-max normalise each list's scores, then blend by dense_weight."""
    sparse_norm = _min_max([r.score for r in sparse])
    dense_norm = _min_max([r.score for r in dense])
    combined: dict[str, float] = {}
    chunk_of: dict[str, Retrieved] = {}
    for i, r in enumerate(sparse):
        cid = r.chunk.chunk_id
        combined[cid] = combined.get(cid, 0.0) + (1 - dense_weight) * sparse_norm[i]
        chunk_of[cid] = r
    for i, r in enumerate(dense):
        cid = r.chunk.chunk_id
        combined[cid] = combined.get(cid, 0.0) + dense_weight * dense_norm[i]
        chunk_of[cid] = r
    ordered = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        Retrieved(chunk=chunk_of[cid].chunk, score=score, rank=rank)
        for rank, (cid, score) in enumerate(ordered, start=1)
    ]


# ---------------------------------------------------------------------------
# 9. RETRIEVAL — STATIC HYBRID (fixed fusion weight)
# ---------------------------------------------------------------------------

class HybridRetriever(Retriever):
    """BM25 + dense with a fixed fusion rule (RRF or weighted)."""

    def __init__(
        self,
        chunks: list[Chunk],
        sparse: BM25Retriever,
        dense: DenseRetriever,
        method: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        fetch_k: int = 20,
    ):
        super().__init__(chunks)
        self.sparse = sparse
        self.dense = dense
        self.method = method
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.fetch_k = fetch_k

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        fetch = max(self.fetch_k, k)
        s = self.sparse.retrieve(query, fetch)
        d = self.dense.retrieve(query, fetch)
        if self.method == "rrf":
            return reciprocal_rank_fusion(s, d, k, rrf_k=self.rrf_k)
        return weighted_fusion(s, d, k, dense_weight=self.dense_weight)


# ---------------------------------------------------------------------------
# 10. RETRIEVAL — ADAPTIVE HYBRID (RQ3 dissertation contribution)
# ---------------------------------------------------------------------------
#
# A transparent, heuristic (untrained) router classifies each query as
# lexical / semantic / mixed using 7 features, then adjusts the sparse-dense
# fusion weight continuously in [min_dense_weight, max_dense_weight].
#
# Features:
#   numeric_token_ratio  — digit / section tokens   -> lexical signal
#   n_quoted_phrases     — "exact phrase" intent    -> lexical signal
#   vocab_overlap        — query tokens in corpus   -> lexical signal
#   length_signal        — short queries = lexical  -> dual signal
#   n_semantic_cues      — why/how/explain/compare  -> semantic signal
#   stop_ratio           — fluent prose = semantic  -> semantic signal
#   retrieval_confidence — which retriever is more decisive

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "as", "by", "at", "from", "is", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "this", "that", "these", "those", "it", "its",
    "what", "which", "who", "whom", "whose", "when", "where", "there", "here",
    "than", "then", "so", "such", "into", "about", "over", "under", "between",
}
_SEMANTIC_CUES = {
    "why", "how", "explain", "describe", "compare", "comparison", "difference",
    "summarize", "summary", "overview", "impact", "reason", "purpose", "benefit",
    "advantage", "disadvantage", "relationship", "effect", "meaning", "interpret",
    "rationale", "implication",
}
_QUOTE_RE = re.compile(r'"[^"]+"')
_SECTION_RE = re.compile(
    r"\b(section|clause|article|appendix|figure|table|step)\b", re.I
)


@dataclass
class RoutingDecision:
    """All features + derived weight — logged in full for reproducibility."""
    label: str              # lexical | semantic | mixed
    dense_weight: float
    semantic_score: float
    lexical_score: float
    n_tokens: int
    numeric_token_ratio: float
    n_quoted_phrases: int
    n_semantic_cues: int
    stopword_ratio: float
    vocab_overlap: float
    retrieval_confidence: float

    def as_log(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveParams:
    min_dense_weight: float = 0.2
    max_dense_weight: float = 0.8


class AdaptiveRetriever(Retriever):
    """
    Heuristic adaptive retriever (RQ3).

    For each query, compute lexical_score and semantic_score from 7 features,
    derive semantic_frac = semantic / (semantic + lexical), then:
        dense_weight = min_dw + semantic_frac * (max_dw - min_dw)
    Apply weighted_fusion with the per-query dense_weight.
    Every decision is stored in self.last_decision for audit logging.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        sparse: BM25Retriever,
        dense: DenseRetriever,
        params: AdaptiveParams,
        fetch_k: int = 20,
    ):
        super().__init__(chunks)
        self.sparse = sparse
        self.dense = dense
        self.params = params
        self.fetch_k = fetch_k
        self._vocab: set[str] = set()
        for c in chunks:
            self._vocab.update(tokenize(c.text))
        self.last_decision: RoutingDecision | None = None

    @staticmethod
    def _retrieval_confidence(ranked: list[Retrieved]) -> float:
        """Normalised top-1 vs top-2 score margin in [0, 1]."""
        if len(ranked) < 2:
            return 1.0 if ranked else 0.0
        top, second = ranked[0].score, ranked[1].score
        denom = abs(top) + abs(second) + 1e-9
        return max(0.0, min(1.0, (top - second) / denom))

    def route(
        self,
        query: str,
        sparse_results: list[Retrieved] | None = None,
        dense_results: list[Retrieved] | None = None,
    ) -> RoutingDecision:
        toks = tokenize(query)
        n = len(toks)

        numeric = sum(1 for t in toks if any(ch.isdigit() for ch in t))
        numeric_ratio = numeric / n if n else 0.0
        if _SECTION_RE.search(query):
            numeric_ratio = min(1.0, numeric_ratio + 0.2)

        n_quoted = len(_QUOTE_RE.findall(query))
        n_cues = sum(1 for t in toks if t in _SEMANTIC_CUES)
        stop_ratio = (sum(1 for t in toks if t in _STOPWORDS) / n) if n else 0.0
        vocab_overlap = (sum(1 for t in toks if t in self._vocab) / n) if n else 0.0

        sparse_conf = self._retrieval_confidence(sparse_results or [])
        dense_conf = self._retrieval_confidence(dense_results or [])
        # positive -> dense more decisive; negative -> sparse more decisive
        retrieval_confidence = dense_conf - sparse_conf

        length_signal = min(n / 20.0, 1.0)   # saturates at 20 tokens

        lexical_score = (
            0.9 * numeric_ratio
            + 0.6 * min(n_quoted, 1)
            + 0.5 * vocab_overlap
            + 0.3 * (1.0 - length_signal)
            + 0.3 * max(0.0, -retrieval_confidence)
        )
        semantic_score = (
            0.7 * min(n_cues, 2) / 2.0
            + 0.5 * length_signal
            + 0.4 * stop_ratio
            + 0.3 * (1.0 - vocab_overlap)
            + 0.3 * max(0.0, retrieval_confidence)
        )

        total = lexical_score + semantic_score + 1e-9
        semantic_frac = semantic_score / total

        p = self.params
        dense_weight = p.min_dense_weight + semantic_frac * (
            p.max_dense_weight - p.min_dense_weight
        )

        margin = semantic_score - lexical_score
        label = "semantic" if margin > 0.15 else ("lexical" if margin < -0.15 else "mixed")

        decision = RoutingDecision(
            label=label,
            dense_weight=round(dense_weight, 4),
            semantic_score=round(semantic_score, 4),
            lexical_score=round(lexical_score, 4),
            n_tokens=n,
            numeric_token_ratio=round(numeric_ratio, 4),
            n_quoted_phrases=n_quoted,
            n_semantic_cues=n_cues,
            stopword_ratio=round(stop_ratio, 4),
            vocab_overlap=round(vocab_overlap, 4),
            retrieval_confidence=round(retrieval_confidence, 4),
        )
        self.last_decision = decision
        return decision

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        fetch = max(self.fetch_k, k)
        s = self.sparse.retrieve(query, fetch)
        d = self.dense.retrieve(query, fetch)
        decision = self.route(query, sparse_results=s, dense_results=d)
        return weighted_fusion(s, d, k, dense_weight=decision.dense_weight)


# ---------------------------------------------------------------------------
# 11. CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    name: str
    retriever: str          # bm25 | dense | hybrid | adaptive
    citations: bool
    top_k: int = 5
    retrieve_k: int = 20
    chunk_size: int = 512
    chunk_overlap: int = 64
    gen_model: str = "gpt-4.1"
    judge_model: str = "gpt-4o"          # different model from generator (gpt-4.1) to reduce self-evaluation bias
    # Local dense retrieval per the proposal (all-MiniLM-L6-v2). Runs on the
    # system interpreter where sentence-transformers loads cleanly; keeps dense
    # index-building offline and free, and removes the network dependency that
    # otherwise crashes the study when the connection blips.
    embed_model: str = "all-MiniLM-L6-v2"
    embed_backend: str = "sentence-transformers"
    judge: str = "openai"
    judge_embed_model: str = "text-embedding-3-large"
    temperature: float = 0.0
    hybrid_method: str = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.5
    adaptive: AdaptiveParams = field(default_factory=AdaptiveParams)


# The eight dissertation configurations.
#
# RQ1: sparse_bm25 / dense  vs  static_hybrid  (RRF fusion — standard in literature)
# RQ2: static_hybrid  vs  static_hybrid_cite  (citation grounding, same retriever)
#      adaptive_hybrid  vs  adaptive_hybrid_cite
# RQ3: static_hybrid_weighted  vs  adaptive_hybrid  (both use weighted_fusion,
#      only difference is fixed weight=0.5 vs per-query adaptive weight)
DISSERTATION_CONFIGS: list[RunConfig] = [
    RunConfig(name="sparse_bm25",                retriever="bm25",     citations=False),
    RunConfig(name="dense",                      retriever="dense",    citations=False),
    RunConfig(name="static_hybrid",              retriever="hybrid",   citations=False),
    RunConfig(name="static_hybrid_cite",         retriever="hybrid",   citations=True),
    RunConfig(name="static_hybrid_weighted",     retriever="hybrid",   citations=False,
              hybrid_method="weighted", dense_weight=0.5),
    RunConfig(name="static_hybrid_weighted_cite",retriever="hybrid",   citations=True,
              hybrid_method="weighted", dense_weight=0.5),
    RunConfig(name="adaptive_hybrid",            retriever="adaptive", citations=False),
    RunConfig(name="adaptive_hybrid_cite",       retriever="adaptive", citations=True),
]

_CONFIG_BY_NAME = {c.name: c for c in DISSERTATION_CONFIGS}


def get_config(name: str) -> RunConfig:
    if name not in _CONFIG_BY_NAME:
        raise KeyError(f"Unknown config {name!r}. Available: {sorted(_CONFIG_BY_NAME)}")
    return _CONFIG_BY_NAME[name]


# ---------------------------------------------------------------------------
# 12. RETRIEVER FACTORY  (builds index once, shares across configs)
# ---------------------------------------------------------------------------

class RetrieverBundle:
    """Builds sparse + dense indexes once per corpus; reuses across all configs."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder | None):
        self.chunks = chunks
        self.embedder = embedder
        self._bm25: BM25Retriever | None = None
        self._dense: DenseRetriever | None = None

    @property
    def bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            self._bm25 = BM25Retriever(self.chunks)
        return self._bm25

    @property
    def dense(self) -> DenseRetriever:
        if self._dense is None:
            if self.embedder is None:
                raise RuntimeError("Dense retrieval requires an embedder.")
            self._dense = DenseRetriever(self.chunks, self.embedder)
        return self._dense

    def for_config(self, cfg: RunConfig) -> Retriever:
        if cfg.retriever == "bm25":
            return self.bm25
        if cfg.retriever == "dense":
            return self.dense
        if cfg.retriever == "hybrid":
            return HybridRetriever(
                self.chunks, self.bm25, self.dense,
                method=cfg.hybrid_method, rrf_k=cfg.rrf_k,
                dense_weight=cfg.dense_weight, fetch_k=cfg.retrieve_k,
            )
        if cfg.retriever == "adaptive":
            return AdaptiveRetriever(
                self.chunks, self.bm25, self.dense,
                params=cfg.adaptive, fetch_k=cfg.retrieve_k,
            )
        raise ValueError(f"Unknown retriever: {cfg.retriever!r}")


# ---------------------------------------------------------------------------
# 13. PROMPT TEMPLATES
# ---------------------------------------------------------------------------

SYSTEM_PLAIN = (
    "You are an enterprise question-answering assistant. Answer the user's "
    "question using ONLY the information in the provided context passages. "
    "If the answer is not contained in the context, reply exactly: "
    "'I don't have enough information to answer that.' "
    "Be concise and factual; do not speculate or use outside knowledge."
)

SYSTEM_CITED = (
    "You are an enterprise question-answering assistant. Answer the user's "
    "question using ONLY the information in the provided numbered context "
    "passages. Every sentence in your answer MUST end with one or more citation "
    "markers in square brackets identifying the passage(s) that support it, e.g. "
    "'Revenue rose 12%. [2]' or 'The clause permits termination. [1][3]'. "
    "If the answer is not contained in the context, reply exactly: "
    "'I don't have enough information to answer that.' "
    "Do not cite passages that do not support the claim. "
    "Do not use outside knowledge."
)


def format_context(passages: list[Retrieved]) -> str:
    lines = []
    for i, r in enumerate(passages, start=1):
        label = r.chunk.title or r.chunk.doc_id
        lines.append(f"[{i}] (source: {label})\n{r.chunk.text}")
    return "\n\n".join(lines)


def build_user_prompt(question: str, passages: list[Retrieved]) -> str:
    return (
        f"Context passages:\n\n{format_context(passages)}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


# ---------------------------------------------------------------------------
# 14. CITATION PARSING & METRICS  (RQ2)
# ---------------------------------------------------------------------------

_MARKER_RE = re.compile(r"\[(\d+)\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_LEADING_MARKERS_RE = re.compile(r"^((?:\[\d+\]\s*)+)")
_REFUSAL = "i don't have enough information"


def _sentences(text: str) -> list[str]:
    """Split on sentence boundaries, reattaching orphaned [n] markers."""
    raw = [s for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    merged: list[str] = []
    for frag in raw:
        m = _LEADING_MARKERS_RE.match(frag)
        if m and merged:
            merged[-1] = f"{merged[-1]} {m.group(1).strip()}".strip()
            rest = frag[m.end():].strip()
            if rest:
                merged.append(rest)
        else:
            merged.append(frag.strip())
    return merged


def citation_coverage(text: str, n_passages: int) -> float:
    """Fraction of answer sentences with at least one valid [n] marker."""
    stripped = text.strip()
    if not stripped or _REFUSAL in stripped.lower():
        return float("nan")
    sents = _sentences(stripped)
    if not sents:
        return float("nan")
    cited = sum(
        1 for s in sents
        if any(1 <= int(m) <= n_passages for m in _MARKER_RE.findall(s))
    )
    return cited / len(sents)


def invalid_citation_rate(text: str, n_passages: int) -> float:
    """Fraction of emitted markers that are out-of-range (hallucinated)."""
    markers = [int(m) for m in _MARKER_RE.findall(text)]
    if not markers:
        return float("nan")
    return sum(1 for i in markers if not (1 <= i <= n_passages)) / len(markers)


def cited_chunk_ids(text: str, passages: list[Retrieved]) -> list[str]:
    idxs = sorted(
        {int(m) for m in _MARKER_RE.findall(text)} & set(range(1, len(passages) + 1))
    )
    return [passages[i - 1].chunk.chunk_id for i in idxs]


# ---------------------------------------------------------------------------
# 15. ANSWER GENERATOR
# ---------------------------------------------------------------------------

class Generator:
    def __init__(self, llm: ChatLLM, temperature: float = 0.0):
        self.llm = llm
        self.temperature = temperature

    def generate(
        self, question: str, passages: list[Retrieved], *, citations: bool
    ) -> GeneratedAnswer:
        system = SYSTEM_CITED if citations else SYSTEM_PLAIN
        user = build_user_prompt(question, passages)
        raw = self.llm.complete(system, user, temperature=self.temperature)
        if citations:
            n = len(passages)
            return GeneratedAnswer(
                text=raw, raw=raw,
                citations=cited_chunk_ids(raw, passages),
                citation_coverage=citation_coverage(raw, n),
                invalid_citation_rate=invalid_citation_rate(raw, n),
            )
        return GeneratedAnswer(text=raw, raw=raw)


# ---------------------------------------------------------------------------
# 16. END-TO-END PIPELINE
# ---------------------------------------------------------------------------

@dataclass
class PipelineOutput:
    question: str
    retrieved: list[Retrieved]
    answer: GeneratedAnswer


class RAGPipeline:
    def __init__(
        self,
        docs: list[Document],
        cfg: RunConfig,
        llm: ChatLLM,
        embedder: Embedder | None,
        bundle: RetrieverBundle | None = None,
    ):
        self.cfg = cfg
        self.chunks = chunk_corpus(docs, cfg.chunk_size, cfg.chunk_overlap)
        self.bundle = bundle or RetrieverBundle(self.chunks, embedder)
        self.retriever = self.bundle.for_config(cfg)
        self.generator = Generator(llm, temperature=cfg.temperature)

    def run(self, question: str) -> PipelineOutput:
        retrieved = self.retriever.retrieve(question, self.cfg.top_k)
        answer = self.generator.generate(
            question, retrieved, citations=self.cfg.citations
        )
        return PipelineOutput(question=question, retrieved=retrieved, answer=answer)


# ---------------------------------------------------------------------------
# 17. RETRIEVAL METRICS  (Recall@K, Precision@K, MRR, nDCG@K, MAP@K)
# ---------------------------------------------------------------------------
#
# NaN-not-zero convention: when a question has no gold labels, every metric
# returns nan so np.nanmean excludes it without biasing the dataset mean.
# Relevance is doc-level: chunk is relevant if chunk.doc_id in gold set.

NAN = float("nan")


def recall_at_k(retrieved: list[Retrieved], gold: set[str], k: int) -> float:
    if not gold:
        return NAN
    hit = {r.chunk.doc_id for r in retrieved[:k] if r.chunk.doc_id in gold}
    return len(hit) / len(gold)


def precision_at_k(retrieved: list[Retrieved], gold: set[str], k: int) -> float:
    if not gold:
        return NAN
    top = retrieved[:k]
    if not top:
        return 0.0
    return len({r.chunk.doc_id for r in top if r.chunk.doc_id in gold}) / k


def reciprocal_rank(retrieved: list[Retrieved], gold: set[str]) -> float:
    if not gold:
        return NAN
    for r in retrieved:
        if r.chunk.doc_id in gold:
            return 1.0 / r.rank
    return 0.0


def ndcg_at_k(retrieved: list[Retrieved], gold: set[str], k: int) -> float:
    if not gold:
        return NAN
    flags = [1 if r.chunk.doc_id in gold else 0 for r in retrieved[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(flags))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved: list[Retrieved], gold: set[str], k: int) -> float:
    if not gold:
        return NAN
    seen: set[str] = set()
    hits = 0
    precisions = []
    for i, r in enumerate(retrieved[:k], start=1):
        if r.chunk.doc_id in gold and r.chunk.doc_id not in seen:
            seen.add(r.chunk.doc_id)
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    return sum(precisions) / min(len(gold), k)


def retrieval_scores(
    retrieved: list[Retrieved], gold_doc_ids: list[str], k: int
) -> dict[str, float]:
    gold = set(gold_doc_ids)
    return {
        f"recall@{k}": recall_at_k(retrieved, gold, k),
        f"precision@{k}": precision_at_k(retrieved, gold, k),
        "mrr": reciprocal_rank(retrieved, gold),
        f"ndcg@{k}": ndcg_at_k(retrieved, gold, k),
        f"map@{k}": average_precision(retrieved, gold, k),
    }


# ---------------------------------------------------------------------------
# 18. RAGAS ANSWER-QUALITY EVALUATION
# ---------------------------------------------------------------------------

_RAGAS_REF_METRICS = {"context_recall", "answer_correctness"}


def evaluate_ragas(
    records: list[dict],
    *,
    judge_provider: str = "openai",
    judge_model: str = "gpt-4o",
    judge_embed_model: str = "text-embedding-3-large",
):
    """
    Score QA records with RAGAS 0.2.x.

    Metrics:
      faithfulness       — answer is grounded in retrieved context
      answer_relevancy   — answer addresses the question (needs embedding)
      context_recall     — retrieved context covers the reference (needs reference)
      answer_correctness — answer matches the reference (needs reference)

    Each record must have: qid, user_input, response, retrieved_contexts, reference.
    Returns a DataFrame with one row per record.
    """
    import pandas as pd
    if not records:
        return pd.DataFrame()

    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        answer_correctness, answer_relevancy, context_recall, faithfulness,
    )
    from langchain_openai import OpenAIEmbeddings

    all_metrics = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }
    judge = make_judge_llm(judge_provider, judge_model)
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=judge_embed_model)
    )
    any_ref = any(r.get("reference") for r in records)
    active_names = [
        n for n in all_metrics if any_ref or n not in _RAGAS_REF_METRICS
    ]
    active_metrics = [all_metrics[n] for n in active_names]

    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=list(r.get("retrieved_contexts") or []),
            reference=r.get("reference"),
        )
        for r in records
    ]
    from ragas.run_config import RunConfig as _RagasRunConfig
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=active_metrics,
        llm=judge,
        embeddings=embeddings,
        # 4 workers instead of the default 16: tier-limited accounts (30k TPM)
        # thrash on 429 retries at higher concurrency.
        run_config=_RagasRunConfig(max_workers=4, max_retries=15, max_wait=90),
    )
    df = result.to_pandas().reset_index(drop=True)
    df.insert(0, "qid", [r["qid"] for r in records])
    for i, r in enumerate(records):
        if not r.get("reference"):
            for m in _RAGAS_REF_METRICS:
                if m in df.columns:
                    df.at[i, m] = math.nan
    return df


# ---------------------------------------------------------------------------
# 19. STATISTICAL ANALYSIS  (Wilcoxon, Cohen's d_z, Holm-Bonferroni)
# ---------------------------------------------------------------------------

@dataclass
class PairedResult:
    metric: str
    config_a: str
    config_b: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float      # b - a  (positive means b wins)
    wilcoxon_p: float
    shapiro_p: float
    normal: bool
    ttest_p: float
    cohens_dz: float
    ci_low: float
    ci_high: float
    p_adjusted: float = float("nan")
    significant: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _clean_pairs(a, b) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]


def bootstrap_ci(
    diff: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    if diff.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def cohens_dz(diff: np.ndarray) -> float:
    if diff.size < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def paired_compare(
    a, b, metric: str, config_a: str, config_b: str
) -> PairedResult:
    """Wilcoxon signed-rank + paired t-test + bootstrap CI + Cohen's d_z."""
    from scipy import stats as sps
    a, b = _clean_pairs(a, b)
    n = a.size
    diff = b - a
    wilcoxon_p = shapiro_p = ttest_p = float("nan")
    if n >= 1:
        if np.allclose(diff, 0):
            wilcoxon_p = ttest_p = 1.0
        else:
            try:
                wilcoxon_p = float(sps.wilcoxon(diff, zero_method="wilcox").pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            if n >= 3:
                shapiro_p = float(sps.shapiro(diff).pvalue)
            try:
                ttest_p = float(sps.ttest_rel(b, a).pvalue)
            except Exception:
                pass
    lo, hi = bootstrap_ci(diff)
    return PairedResult(
        metric=metric, config_a=config_a, config_b=config_b, n=int(n),
        mean_a=float(np.nanmean(a)) if n else NAN,
        mean_b=float(np.nanmean(b)) if n else NAN,
        mean_diff=float(diff.mean()) if n else NAN,
        wilcoxon_p=wilcoxon_p, shapiro_p=shapiro_p,
        normal=(shapiro_p > 0.05) if not math.isnan(shapiro_p) else False,
        ttest_p=ttest_p, cohens_dz=cohens_dz(diff), ci_low=lo, ci_high=hi,
    )


def holm_bonferroni(
    results: list[PairedResult], alpha: float = 0.05, use: str = "wilcoxon"
) -> list[PairedResult]:
    """Holm-Bonferroni FWER correction, applied in-place."""
    key = (lambda r: r.wilcoxon_p) if use == "wilcoxon" else (lambda r: r.ttest_p)
    valid = [r for r in results if not math.isnan(key(r))]
    m = len(valid)
    order = sorted(valid, key=key)
    for rank, r in enumerate(order):
        r.p_adjusted = float(min(1.0, (m - rank) * key(r)))
        r.significant = r.p_adjusted < alpha
    running = 0.0
    for r in order:
        running = max(running, r.p_adjusted)
        r.p_adjusted = running
        r.significant = r.p_adjusted < alpha
    return results


def default_comparisons(metric_cols: list[str]) -> list[tuple]:
    """
    RQ1-RQ3 comparison pairs.

    RQ1 — hybrid vs baselines: uses static_hybrid (RRF), the canonical hybrid in
          the retrieval literature (Cormack & Clarke 2009).
    RQ2 — citation grounding: same retriever, ±citations.
    RQ3 — adaptive routing: both sides use weighted_fusion; the ONLY difference is
          fixed dense_weight=0.5 (static_hybrid_weighted) vs per-query adaptive
          weight (adaptive_hybrid).  This controls the fusion-method confound.
    """
    retrieval = [
        c for c in metric_cols
        if c.startswith(("recall@", "precision@", "ndcg@", "map@")) or c == "mrr"
    ]
    quality = [
        c for c in ["faithfulness", "answer_relevancy", "answer_correctness", "context_recall"]
        if c in metric_cols
    ]
    faith = [c for c in ["faithfulness", "answer_correctness"] if c in metric_cols]
    # `quality` already contains answer_relevancy; the RQ3 lists use `retrieval +
    # quality` so each metric is tested exactly once and the Holm-Bonferroni family
    # size per dataset is not inflated by a duplicated answer_relevancy comparison.
    return [
        ("RQ1", "sparse_bm25",               "static_hybrid",               retrieval),
        ("RQ1", "dense",                      "static_hybrid",               retrieval),
        ("RQ2", "static_hybrid",              "static_hybrid_cite",          faith),
        ("RQ2", "adaptive_hybrid",            "adaptive_hybrid_cite",        faith),
        ("RQ3", "static_hybrid_weighted",     "adaptive_hybrid",             retrieval + quality),
        ("RQ3", "static_hybrid_weighted_cite","adaptive_hybrid_cite",        retrieval + quality),
    ]


def run_rq_analysis(df, metric_cols: list[str] | None = None, alpha: float = 0.05):
    """Run RQ1-RQ3 paired comparisons per dataset; return a tidy DataFrame."""
    import pandas as pd
    if metric_cols is None:
        skip = {
            "dataset", "config", "qid", "question", "answer",
            "retrieved_doc_ids", "n_retrieved", "route_label", "route_dense_weight",
        }
        metric_cols = [
            c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])
        ]
    rows: list[PairedResult] = []
    for dataset, dsub in df.groupby("dataset"):
        ds_results: list[PairedResult] = []
        for rq, base, treat, metrics in default_comparisons(metric_cols):
            for metric in metrics:
                if metric not in dsub.columns:
                    continue
                a = dsub[dsub.config == base].set_index("qid")[metric]
                b = dsub[dsub.config == treat].set_index("qid")[metric]
                if a.empty or b.empty:
                    continue
                common = a.index.intersection(b.index)
                if not len(common):
                    continue
                res = paired_compare(
                    a.loc[common].values, b.loc[common].values,
                    metric, base, treat,
                )
                res._rq = rq            # type: ignore[attr-defined]
                res._dataset = dataset  # type: ignore[attr-defined]
                ds_results.append(res)
        holm_bonferroni(ds_results, alpha=alpha)
        rows.extend(ds_results)
    records = [
        {**r.as_dict(), "rq": getattr(r, "_rq", ""), "dataset": getattr(r, "_dataset", "")}
        for r in rows
    ]
    cols = [
        "rq", "dataset", "metric", "config_a", "config_b", "n",
        "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high",
        "cohens_dz", "wilcoxon_p", "shapiro_p", "normal", "ttest_p",
        "p_adjusted", "significant",
    ]
    out = pd.DataFrame(records)
    return out[cols] if not out.empty else out


# ---------------------------------------------------------------------------
# 20. DATASET LOADERS  (JSONL convention)
# ---------------------------------------------------------------------------

_DATA_DIR = Path(
    os.environ.get("AHRAG_DATA_DIR", Path(__file__).parent / "data")
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def docs_from_jsonl(path: Path) -> list[Document]:
    return [
        Document(
            doc_id=str(r["doc_id"]),
            text=r["text"],
            title=r.get("title", ""),
            metadata=r.get("metadata", {}) or {},
        )
        for r in _read_jsonl(path)
    ]


def qa_from_jsonl(path: Path) -> list[QAItem]:
    return [
        QAItem(
            qid=str(r["qid"]),
            question=r["question"],
            reference_answer=r.get("reference_answer"),
            gold_doc_ids=[str(x) for x in (r.get("gold_doc_ids") or [])],
            metadata=r.get("metadata", {}) or {},
        )
        for r in _read_jsonl(path)
    ]


def load_dataset(name: str) -> tuple[list[Document], list[QAItem]]:
    """
    Load a named dataset from data/<name>/corpus.jsonl + qa.jsonl.

    Built-in: sample (ships in data/sample/)
    External: financebench, cuad, azure (place JSONL files under data/<name>/)
    """
    base = _DATA_DIR / name
    corpus_path = base / "corpus.jsonl"
    qa_path = base / "qa.jsonl"
    if not corpus_path.exists() or not qa_path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found.\n"
            f"Expected files:\n  {corpus_path}\n  {qa_path}\n"
            "Place corpus.jsonl and qa.jsonl in that folder and re-run."
        )
    return docs_from_jsonl(corpus_path), qa_from_jsonl(qa_path)


# ---------------------------------------------------------------------------
# 21. PER-CONFIG EXPERIMENT RUNNER
# ---------------------------------------------------------------------------

def _audit_record(item: QAItem, out: PipelineOutput, cfg: RunConfig) -> dict:
    """Full reconstructable trace for one query (written to JSONL)."""
    return {
        "config": cfg.name,
        "qid": item.qid,
        "question": item.question,
        "reference_answer": item.reference_answer,
        "gold_doc_ids": item.gold_doc_ids,
        "retrieved": [
            {
                "rank": r.rank,
                "score": round(float(r.score), 6),
                "chunk_id": r.chunk.chunk_id,
                "doc_id": r.chunk.doc_id,
            }
            for r in out.retrieved
        ],
        "prompt": build_user_prompt(item.question, out.retrieved),
        "answer": out.answer.text,
        "citations": out.answer.citations,
        "citation_coverage": out.answer.citation_coverage,
        "invalid_citation_rate": out.answer.invalid_citation_rate,
    }


def run_experiment(
    docs: list[Document],
    items: list[QAItem],
    cfg: RunConfig,
    llm: ChatLLM,
    embedder: Embedder | None,
    *,
    dataset_name: str = "",
    use_ragas: bool = True,
    bundle: RetrieverBundle | None = None,
    progress: bool = True,
    audit_path: str | Path | None = None,
):
    """
    Run one RunConfig over one dataset.

    Returns a per-question DataFrame with retrieval metrics, answer quality
    scores (RAGAS), citation metrics, and routing features (adaptive only).
    Writes a JSONL audit log if audit_path is supplied.
    """
    import pandas as pd
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **_): return it  # type: ignore

    pipeline = RAGPipeline(docs, cfg, llm, embedder, bundle=bundle)
    is_adaptive = isinstance(pipeline.retriever, AdaptiveRetriever)

    rows: list[dict] = []
    ragas_records: list[dict] = []
    audit_records: list[dict] = []

    for item in tqdm(items, desc=f"{dataset_name}:{cfg.name}", disable=not progress):
        out = pipeline.run(item.question)
        retr = retrieval_scores(out.retrieved, item.gold_doc_ids, cfg.top_k)

        row: dict = {
            "dataset": dataset_name,
            "config": cfg.name,
            "qid": item.qid,
            "question": item.question,
            "answer": out.answer.text,
            "citation_coverage": out.answer.citation_coverage,
            "invalid_citation_rate": out.answer.invalid_citation_rate,
            "n_retrieved": len(out.retrieved),
            "retrieved_doc_ids": "|".join(r.chunk.doc_id for r in out.retrieved),
        }
        row.update(retr)

        audit = _audit_record(item, out, cfg)
        if is_adaptive and pipeline.retriever.last_decision is not None:
            decision = pipeline.retriever.last_decision.as_log()
            row["route_label"] = decision["label"]
            row["route_dense_weight"] = decision["dense_weight"]
            audit["routing"] = decision

        rows.append(row)
        audit_records.append(audit)
        ragas_records.append({
            "qid": item.qid,
            "user_input": item.question,
            "response": out.answer.text,
            "retrieved_contexts": [r.chunk.text for r in out.retrieved],
            "reference": item.reference_answer,
        })

    df = pd.DataFrame(rows)

    if audit_path is not None:
        ap = Path(audit_path)
        ap.parent.mkdir(parents=True, exist_ok=True)
        with open(ap, "w", encoding="utf-8") as fh:
            for rec in audit_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if use_ragas:
        ragas_df = evaluate_ragas(
            ragas_records,
            judge_provider=cfg.judge,
            judge_model=cfg.judge_model,
            judge_embed_model=cfg.judge_embed_model,
        )
        if not ragas_df.empty:
            drop = {"user_input", "response", "retrieved_contexts", "reference"}
            keep = [c for c in ragas_df.columns if c not in drop]
            df = df.merge(ragas_df[keep], on="qid", how="left")

    return df


def save_results(
    df, results_dir: str | Path, dataset: str, config: str
) -> Path:
    out_dir = Path(results_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{config}.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 22. STUDY ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_study(
    datasets: list[str],
    configs: list[RunConfig],
    *,
    results_dir: str = "results",
    use_ragas: bool = True,
    limit: int | None = None,
    progress: bool = True,
):
    """
    Run all configs over all datasets.

    Builds the BM25 and FAISS indexes ONCE per dataset, reuses them across
    all configs. Writes per-config CSVs + combined all_results.csv.
    """
    import pandas as pd

    llm = make_generator(configs[0].gen_model)
    embedder = make_dense_embedder(configs[0].embed_model, configs[0].embed_backend)
    all_frames: list[pd.DataFrame] = []

    for ds_name in datasets:
        print(f"\n[Dataset: {ds_name}]")
        docs, items = load_dataset(ds_name)
        if limit:
            items = items[:limit]
        print(f"  {len(docs)} docs | {len(items)} questions")

        base = configs[0]
        chunks = chunk_corpus(docs, base.chunk_size, base.chunk_overlap)
        bundle = RetrieverBundle(chunks, embedder)
        print(f"  {len(chunks)} chunks indexed")

        for cfg in configs:
            done_path = Path(results_dir) / ds_name / f"{cfg.name}.csv"
            if done_path.exists():
                df = pd.read_csv(done_path)
                # Only treat a config as complete if its RAGAS coverage is
                # adequate. A network drop mid-config saves a CSV full of NaN
                # judge scores; without this check, resume would skip it forever.
                complete = True
                if use_ragas and "faithfulness" in df.columns and len(df):
                    coverage = df["faithfulness"].notna().mean()
                    complete = coverage >= 0.8
                if complete:
                    print(f"  {cfg.name}: [resume] {len(df)} rows already at {done_path} — skipped")
                    all_frames.append(df)
                    continue
                print(f"  {cfg.name}: [resume] {done_path} has low RAGAS coverage "
                      f"({df['faithfulness'].notna().mean():.0%}) — re-running")
            audit_path = (
                Path(results_dir) / ds_name / "audit" / f"{cfg.name}.jsonl"
            )
            df = run_experiment(
                docs, items, cfg, llm, embedder,
                dataset_name=ds_name,
                use_ragas=use_ragas,
                bundle=bundle,
                progress=progress,
                audit_path=audit_path,
            )
            path = save_results(df, results_dir, ds_name, cfg.name)
            print(f"  {cfg.name}: {len(df)} rows -> {path}")
            all_frames.append(df)

    combined = (
        pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    )
    if not combined.empty:
        out = Path(results_dir) / "all_results.csv"
        combined.to_csv(out, index=False)
        print(f"\nCombined results -> {out}")
    return combined


# ---------------------------------------------------------------------------
# 24. DATASET BUILDERS  (HuggingFace -> local JSONL)
# ---------------------------------------------------------------------------
#
# FinanceBench evidence strings are used as the corpus text (an open-book
# approximation of full-document retrieval — report this as a limitation).
# Azure is a custom corpus and is NOT built here; place its JSONL manually.

def _write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):5d} rows -> {path}")


def build_financebench():
    """
    PatronusAI/financebench -> data/financebench/{corpus,qa}.jsonl

    doc_id is a SHA-256 hash of the evidence text.  Identical evidence passages
    (e.g. the same SEC exhibit quoted by two questions) map to one shared corpus
    document, so the retriever cannot be penalised for returning the "twin" doc
    instead of the nominally different one.  Questions that share evidence text
    share the same gold_doc_id and point at the single canonical document.
    """
    import hashlib
    from datasets import load_dataset as hf_load
    ds = hf_load("PatronusAI/financebench", split="train")
    corpus_seen: dict[str, str] = {}   # text_hash -> doc_id
    corpus, qa = [], []
    for i, ex in enumerate(ds):
        doc_name = ex.get("doc_name") or f"fb-doc-{i}"
        fb_id = str(ex.get("financebench_id") or f"fb-{i}")
        ev = ex.get("evidence")
        context = ""
        if isinstance(ev, list) and ev:
            context = " ".join(
                str(e.get("evidence_text", "")) for e in ev if isinstance(e, dict)
            )
        elif isinstance(ev, str):
            context = ev
        context = context.strip()
        if context:
            text_hash = hashlib.sha256(context.encode()).hexdigest()[:16]
            if text_hash not in corpus_seen:
                doc_id = f"{doc_name}::{text_hash}"
                corpus_seen[text_hash] = doc_id
                corpus.append({"doc_id": doc_id, "title": doc_name, "text": context})
            gold_doc_id = corpus_seen[text_hash]
        else:
            gold_doc_id = None
        qa.append({
            "qid": fb_id,
            "question": ex["question"],
            "reference_answer": ex.get("answer"),
            "gold_doc_ids": [gold_doc_id] if gold_doc_id else [],
        })
    _write_jsonl(corpus, _DATA_DIR / "financebench" / "corpus.jsonl")
    _write_jsonl(qa, _DATA_DIR / "financebench" / "qa.jsonl")


def build_cuad(n_questions: int = 150, seed: int = 42):
    """Real CUAD (Hendrycks et al., 2021) -> data/cuad/{corpus,qa}.jsonl.

    Source: official Atticus repo data.zip -> test.json (SQuAD format).
    The QA data lives inside data.zip at the repo root, not at data/test.json.
    Corpus = all 102 test-split contracts (full text concatenated across
    paragraphs); QA = seeded random sample of `n_questions` answerable
    questions. Set n_questions=None for all ~1,244 answerable questions.
    """
    import urllib.request, zipfile, io, random

    URL = "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data.zip"
    print(f"  Downloading real CUAD from {URL} ...")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
    data = json.load(zf.open("test.json"))["data"]

    corpus, pool, seen = [], [], set()
    for art in data:
        title = art["title"]
        doc_id = re.sub(r"\W+", "_", title)[:60].strip("_") or f"cuad-doc-{len(corpus)}"
        base, k = doc_id, 1
        while doc_id in seen:
            doc_id, k = f"{base}_{k}", k + 1
        seen.add(doc_id)
        corpus.append({
            "doc_id": doc_id,
            "title": title,
            "text": " ".join(p["context"] for p in art["paragraphs"]),
        })
        for p in art["paragraphs"]:
            for q in p["qas"]:
                ans = [a["text"] for a in q.get("answers", []) if a.get("text", "").strip()]
                if ans and not q.get("is_impossible", False):
                    pool.append({
                        "qid": str(q["id"]),
                        "question": q["question"],
                        "reference_answer": ans[0],
                        "gold_doc_ids": [doc_id],
                    })

    random.Random(seed).shuffle(pool)
    _write_jsonl(corpus, _DATA_DIR / "cuad" / "corpus.jsonl")
    _write_jsonl(pool[:n_questions] if n_questions else pool,
                 _DATA_DIR / "cuad" / "qa.jsonl")




def build_dataset(name: str):
    print(f"[build-data] {name} ...")
    if name == "financebench":
        build_financebench()
    elif name == "cuad":
        build_cuad()
    else:
        raise ValueError(
            f"No HuggingFace builder for {name!r}. Azure is a custom corpus — "
            "place data/azure/corpus.jsonl + qa.jsonl manually."
        )


# ---------------------------------------------------------------------------
# 25. ANALYSIS & FIGURES  (summary means, RQ stats, error analysis, charts)
# ---------------------------------------------------------------------------

_METRIC_PREFIXES = ("recall@", "precision@", "ndcg@", "map@")
_SCALAR_METRICS = [
    "mrr", "citation_coverage", "invalid_citation_rate",
    "faithfulness", "answer_relevancy", "context_recall", "answer_correctness",
]
_CONFIG_ORDER = [c.name for c in DISSERTATION_CONFIGS]  # display order for figures


def _metric_columns(df):
    cols = [c for c in df.columns if c.startswith(_METRIC_PREFIXES)]
    cols += [c for c in _SCALAR_METRICS if c in df.columns]
    return cols


def _load_all_results(results_dir):
    import pandas as pd
    rd = Path(results_dir)
    combined = rd / "all_results.csv"
    if combined.exists():
        return pd.read_csv(combined)
    frames = [pd.read_csv(p) for p in rd.glob("*/*.csv")]
    if not frames:
        raise FileNotFoundError(f"No result CSVs under {rd}. Run the study first.")
    return pd.concat(frames, ignore_index=True)


def summarise_results(df, results_dir):
    import numpy as np, pandas as pd
    metrics = _metric_columns(df)
    summary = (df.groupby(["dataset", "config"])[metrics]
                 .agg(lambda s: np.nanmean(pd.to_numeric(s, errors="coerce")))
                 .reset_index())
    out = Path(results_dir) / "summary.csv"
    summary.to_csv(out, index=False)
    print(f"  summary -> {out}")
    return summary


def error_analysis(df, results_dir, faithfulness_threshold=0.7):
    import pandas as pd, numpy as np
    if "faithfulness" not in df.columns:
        print("  [error-analysis skipped] no faithfulness column (run with RAGAS)")
        return
    recall_cols = [c for c in df.columns if c.startswith("recall@")]
    rc = recall_cols[0] if recall_cols else None
    adaptive = {"adaptive_hybrid", "adaptive_hybrid_cite"}
    cited = {"static_hybrid_cite", "adaptive_hybrid_cite"}

    def categorise(row):
        inv = row.get("invalid_citation_rate", np.nan)
        rec = row.get(rc, np.nan) if rc else np.nan
        if row["config"] in cited and pd.notna(inv) and inv > 0:
            return "citation_mismatch"
        if pd.notna(rec) and rec == 0:
            return "routing_error" if row["config"] in adaptive else "retrieval_failure"
        if pd.notna(rec) and rec > 0:
            return "unsupported_synthesis"
        return "undetermined"

    low = df[pd.to_numeric(df["faithfulness"], errors="coerce") < faithfulness_threshold].copy()
    if low.empty:
        print(f"  [error-analysis] no cases below faithfulness {faithfulness_threshold}")
        return
    low["error_category"] = low.apply(categorise, axis=1)
    low.to_csv(Path(results_dir) / "error_analysis.csv", index=False)
    (low.groupby(["dataset", "config", "error_category"]).size()
        .reset_index(name="n")
        .to_csv(Path(results_dir) / "error_analysis_summary.csv", index=False))
    print(f"  error-analysis -> {Path(results_dir) / 'error_analysis_summary.csv'} "
          f"({len(low)} low-faithfulness cases)")


def make_figures(df, results_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [figures skipped] matplotlib not installed")
        return
    import numpy as np, pandas as pd
    fig_dir = Path(results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = _metric_columns(df)
    means = (df.groupby(["dataset", "config"])[metrics]
               .agg(lambda s: np.nanmean(pd.to_numeric(s, errors="coerce"))).reset_index())
    n = 0
    for dataset, dsub in means.groupby("dataset"):
        order = [c for c in _CONFIG_ORDER if c in dsub["config"].values]
        dsub = dsub.set_index("config").reindex(order)
        for metric in metrics:
            if dsub[metric].isna().all():
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5))
            vals = dsub[metric].values
            ax.bar(range(len(dsub.index)), vals, color="#3b6ea5")
            ax.set_xticks(range(len(dsub.index)))
            ax.set_xticklabels(dsub.index, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} by configuration — {dataset}")
            top = np.nanmax(vals)
            ax.set_ylim(0, max(0.001, top) * 1.15)
            for i, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{dataset}__{metric.replace('@', '_at_')}.png", dpi=150)
            plt.close(fig)
            n += 1
    print(f"  figures -> {fig_dir} ({n} charts)")


def run_analysis(results_dir="results", faithfulness_threshold=0.7):
    """Produce summary.csv, stats.csv, error analysis and figures from results/."""
    df = _load_all_results(results_dir)
    print("\n[analysis]")
    summarise_results(df, results_dir)
    try:
        stats = run_rq_analysis(df)
        if not stats.empty:
            stats.to_csv(Path(results_dir) / "stats.csv", index=False)
            print(f"  stats -> {Path(results_dir) / 'stats.csv'}")
    except Exception as e:
        print(f"  [stats skipped] {e}")
    error_analysis(df, results_dir, faithfulness_threshold)
    make_figures(df, results_dir)
    print("[analysis] done.")


# ---------------------------------------------------------------------------
# 26. CLI ENTRY POINT
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    p = argparse.ArgumentParser(
        description="Adaptive Hybrid RAG for Enterprise QA — Dissertation Study Runner"
    )
    p.add_argument("--datasets", nargs="+", default=["sample"],
                   choices=["sample", "financebench", "cuad", "azure"],
                   help="Which datasets to evaluate on (default: sample)")
    p.add_argument("--configs", nargs="+", default=None, metavar="NAME",
                   help="Subset of configs (default: all 8). Choices: "
                        + ", ".join(c.name for c in DISSERTATION_CONFIGS))
    p.add_argument("--results-dir", default="results",
                   help="Directory for output CSVs (default: results/)")
    p.add_argument("--no-ragas", action="store_true",
                   help="Skip RAGAS scoring (faster; no judge API cost)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap questions per dataset (quick tests)")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate data + chunking only — no API keys needed")
    p.add_argument("--build-data", nargs="+", metavar="NAME",
                   choices=["financebench", "cuad"],
                   help="Download FinanceBench/CUAD from HuggingFace into data/<name>/ and exit")
    p.add_argument("--analyze-only", action="store_true",
                   help="Skip running; analyse existing results in --results-dir and exit")
    p.add_argument("--no-analyze", action="store_true",
                   help="Do not auto-run analysis after the study")
    p.add_argument("--faithfulness-threshold", type=float, default=0.7,
                   help="Error-analysis faithfulness cut-off (default 0.7)")
    args = p.parse_args(argv)

    if args.build_data:
        for name in args.build_data:
            build_dataset(name)
        print("[build-data] done.")
        return 0

    if args.analyze_only:
        run_analysis(args.results_dir, args.faithfulness_threshold)
        return 0

    cfg_names = args.configs or [c.name for c in DISSERTATION_CONFIGS]
    try:
        configs = [get_config(n) for n in cfg_names]
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Datasets : {args.datasets}")
    print(f"Configs  : {[c.name for c in configs]}")
    print(f"RAGAS    : {not args.no_ragas}")

    if args.dry_run:
        print("\n[dry-run] Validating datasets and chunking (no API keys needed)...")
        for ds_name in args.datasets:
            docs, items = load_dataset(ds_name)
            if args.limit:
                items = items[:args.limit]
            chunks = chunk_corpus(docs, configs[0].chunk_size, configs[0].chunk_overlap)
            print(f"  {ds_name}: {len(docs)} docs | {len(items)} questions | {len(chunks)} chunks  OK")
        print("[dry-run] All checks passed.")
        return 0

    run_study(
        datasets=args.datasets,
        configs=configs,
        results_dir=args.results_dir,
        use_ragas=not args.no_ragas,
        limit=args.limit,
        progress=not args.no_progress,
    )

    if not args.no_analyze:
        run_analysis(args.results_dir, args.faithfulness_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
