"""Retriever: question → list of relevant chunks.

Right now this is a thin wrapper over the vector store. In Week 3 we'll add
re-ranking, query expansion, and hybrid search here — that's where the
A/B testing happens.
"""

from __future__ import annotations

from src.config import cfg
from src.store import Hit, VectorStore


class Retriever:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or VectorStore()

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        return self.store.query(question, k=k or cfg.top_k)
