"""Cross-encoder reranker.

Pattern:
  bi-encoder retrieves a candidate pool (fast, approximate)
  → cross-encoder reranks the pool (slow, accurate)
  → top-K returned to generator

Why this pattern is everywhere in production:
  Cosine over independent embeddings (bi-encoder) is a *similarity* signal,
  not a *relevance* signal. "Frontend" and "backend" might cosine-cluster
  near each other even when the chunk doesn't actually answer your question.
  A cross-encoder reads (query, chunk) together and outputs a true relevance
  score — much higher quality, but you can't pre-index it. So you over-fetch
  with the cheap retriever, then rerank a small set with the accurate one.

Two model tiers we support:
  miniLM (cheap baseline): cross-encoder/ms-marco-MiniLM-L-6-v2
    ~22M params, 6 transformer layers, ~80MB, ~50ms/pair CPU
    Trained on MS MARCO web search. Cheap, fast, sometimes hurts.
  BGE-large (best in class): BAAI/bge-reranker-large
    ~560M params, ~1.3GB, ~1-3s/pair CPU (or ~20ms on GPU)
    Trained on diverse multi-domain ranking data including code/docs.
    SOTA on most public reranker benchmarks as of 2024.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.store import Hit


# ─── Named rerankers we ship ──────────────────────────────────────────────
RERANKER_MODELS: dict[str, str] = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-large": "BAAI/bge-reranker-large",
}
DEFAULT_MODEL_NAME = RERANKER_MODELS["minilm"]


@dataclass
class RerankConfig:
    model_name: str = DEFAULT_MODEL_NAME
    # If we ever want to tune by content length etc., put it here.


class Reranker:
    """Cross-encoder reranker. Lazy-loads the model on first use so importing
    this file is cheap (good for tests and other code paths that don't need it)."""

    def __init__(self, config: RerankConfig | None = None) -> None:
        self.config = config or RerankConfig()
        self._model = None  # lazy

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        # Heavy import: pulls torch + transformers. Only happens on first rerank.
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.config.model_name)

    def rerank(self, query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
        """Rerank `hits` by cross-encoder relevance to `query`.

        Returns the same Hit objects in new order, with .score replaced by the
        cross-encoder's relevance score (higher = more relevant).
        """
        if not hits:
            return []

        self._ensure_model()
        pairs = [(query, h.text) for h in hits]
        raw = self._model.predict(pairs)  # type: ignore[attr-defined]
        # `predict` returns numpy ndarray in real use; tolerate plain lists too.
        scores = raw.tolist() if hasattr(raw, "tolist") else list(raw)

        ranked = sorted(
            zip(hits, scores),
            key=lambda hs: -hs[1],
        )

        out: list[Hit] = []
        for hit, score in ranked:
            out.append(Hit(
                text=hit.text,
                source=hit.source,
                section=hit.section,
                score=float(score),  # cross-encoder score
                id=hit.id,
            ))

        if top_k is not None:
            out = out[:top_k]
        return out
