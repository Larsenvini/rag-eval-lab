"""Hybrid retrieval: BM25 (keyword) + vector (semantic), fused via RRF.

WHY:
  Vector embeddings can miss queries that hinge on exact-match keywords
  (e.g. "frontend", "backend", YAML field names) when the corpus uses
  different surface forms ("connecting frontend to backend" vs "deploying
  a frontend"). BM25 catches those. Combining the two ranked lists is
  almost always better than either alone.

HOW (Reciprocal Rank Fusion):
  For each candidate chunk c, score(c) = sum over rankers r of:
      1 / (k_rrf + rank_r(c))
  where rank starts at 1 for the top result. We use k_rrf=60 — the
  Cormack et al. (2009) default that's been adopted by Elastic, Vespa,
  and most modern hybrid retrieval systems.

  RRF only cares about *rank position*, not raw scores. That sidesteps
  the BM25-vs-cosine score normalization problem entirely.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.config import cfg
from src.store import Hit, VectorStore


# ─── Tokenizer ────────────────────────────────────────────────────────────

# Lightweight, dependency-free tokenizer. Lowercase, alphanumerics + hyphens
# (so "k8s", "config-map", "node-port" survive as single tokens).
# Drops common English stopwords that hurt BM25 signal.
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]+", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "with", "by", "as", "at", "from", "this", "that",
    "these", "those", "it", "its", "you", "your", "we", "our", "i", "me",
    "what", "which", "how", "why", "when", "where", "who", "can", "could",
    "should", "would", "will", "may", "might", "than", "then",
})


def _tokenize(text: str) -> list[str]:
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOPWORDS
    ]


# ─── Hybrid retriever ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class HybridConfig:
    """Settings for hybrid retrieval. Defaults match the ones we'll baseline."""
    rrf_k: int = 60          # RRF constant; 60 is the standard
    candidate_pool: int = 30  # how many to fetch from each retriever before fusing
    final_k: int | None = None  # final returned count; None = use cfg.top_k

    # Reranking (optional). If a reranker is supplied to HybridRetriever:
    #   - we keep `rerank_pool` items after RRF
    #   - the reranker reorders them
    #   - we return final_k from the top
    rerank_pool: int = 30


class HybridRetriever:
    """BM25 + vector retriever with RRF fusion. Optional cross-encoder rerank.

    Drop-in compatible with Retriever — exposes .retrieve(question, k=None).
    """

    def __init__(
        self,
        store: VectorStore | None = None,
        config: HybridConfig | None = None,
        reranker: object | None = None,    # type: src.reranker.Reranker
    ) -> None:
        self.store = store or VectorStore()
        self.config = config or HybridConfig()
        self.reranker = reranker

        # Build BM25 index once at startup. K8s docs at our scale (~500 chunks)
        # makes this <1s and well under 100MB RAM. Re-index if the store changes.
        self._chunks: list[Hit] = self.store.all_chunks()
        if not self._chunks:
            raise RuntimeError(
                "Vector store is empty. Run `python -m scripts.ingest` first."
            )
        tokenized_corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._chunks_by_id = {c.id: c for c in self._chunks}

    # Public surface — matches Retriever.retrieve()
    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        final_k = k or self.config.final_k or cfg.top_k

        # 1. BM25 ranks
        bm25_ranked = self._bm25_search(question, n=self.config.candidate_pool)

        # 2. Vector ranks (over-fetch to give RRF more to work with)
        vector_ranked = self.store.query(question, k=self.config.candidate_pool)

        # 3. RRF fusion
        fused = self._rrf_fuse([bm25_ranked, vector_ranked])

        # 4. Optional cross-encoder rerank.
        # We hand the reranker the top `rerank_pool` candidates, let it reorder,
        # then take final_k from the top of that.
        if self.reranker is not None:
            pool = fused[: self.config.rerank_pool]
            return self.reranker.rerank(question, pool, top_k=final_k)

        return fused[:final_k]


    # ── internals ────────────────────────────────────────────────────────

    def _bm25_search(self, query: str, n: int) -> list[Hit]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Argsort descending, take top-n with strictly positive score
        ranked_idx = sorted(
            (i for i, s in enumerate(scores) if s > 0),
            key=lambda i: -scores[i],
        )[:n]
        out: list[Hit] = []
        for idx in ranked_idx:
            base = self._chunks[idx]
            # We piggy-back BM25 score on the score field for inspection;
            # downstream code (generator, eval) doesn't depend on its scale.
            out.append(Hit(
                text=base.text,
                source=base.source,
                section=base.section,
                score=float(scores[idx]),  # BM25 score (higher = better)
                id=base.id,
            ))
        return out

    def _rrf_fuse(self, ranked_lists: list[list[Hit]]) -> list[Hit]:
        """Reciprocal Rank Fusion.

        For every chunk that appears in any input list, sum 1/(k + rank).
        Higher fused score = better. Returns merged Hits sorted desc.
        """
        rrf_k = self.config.rrf_k
        scores: dict[str, float] = defaultdict(float)
        # Track the best Hit object per id so we can return texts/sources cleanly
        best_hit: dict[str, Hit] = {}

        for ranked in ranked_lists:
            for rank, hit in enumerate(ranked, start=1):
                if not hit.id:
                    continue  # skip anything malformed
                scores[hit.id] += 1.0 / (rrf_k + rank)
                # Keep the original Hit (vector hit if present, else bm25)
                if hit.id not in best_hit:
                    best_hit[hit.id] = hit

        # Build ordered output. The Hit.score we expose is the RRF score —
        # a number on a known scale (0, ~0.03] that's easy to interpret.
        merged: list[Hit] = []
        for chunk_id, score in sorted(scores.items(), key=lambda kv: -kv[1]):
            base = best_hit[chunk_id]
            merged.append(Hit(
                text=base.text,
                source=base.source,
                section=base.section,
                score=float(score),  # RRF score (higher = better)
                id=base.id,
            ))
        return merged


def build_retriever(mode: str = "dense") -> object:
    """Factory: 'dense', 'hybrid', 'hybrid+rerank', or 'hybrid+rerank-large'.

    Used by run_eval and ask.py so we can switch retrievers from the CLI
    without touching downstream code.
    """
    if mode == "hybrid":
        return HybridRetriever()
    if mode == "hybrid+rerank":
        from src.reranker import Reranker, RerankConfig, RERANKER_MODELS
        return HybridRetriever(reranker=Reranker(RerankConfig(model_name=RERANKER_MODELS["minilm"])))
    if mode == "hybrid+rerank-large":
        from src.reranker import Reranker, RerankConfig, RERANKER_MODELS
        return HybridRetriever(reranker=Reranker(RerankConfig(model_name=RERANKER_MODELS["bge-large"])))
    if mode == "dense":
        from src.retriever import Retriever
        return Retriever()
    raise ValueError(f"Unknown retriever mode: {mode!r}")
