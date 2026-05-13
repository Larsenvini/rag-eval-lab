"""ChromaDB wrapper. Hides the vector-DB details from the rest of the app so we
can swap to Pinecone/Qdrant later without touching retriever code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from tqdm import tqdm

from src.chunker import Chunk
from src.config import cfg


@dataclass
class Hit:
    """A retrieval result."""
    text: str
    source: str
    section: str
    score: float    # distance — lower is closer for cosine in Chroma
    id: str = ""    # chroma id (e.g. "concepts/workloads/pods.md::3"); used for dedup/fusion


class VectorStore:
    """Thin wrapper around a Chroma collection."""

    def __init__(self) -> None:
        cfg.chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(cfg.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._openai = OpenAI(api_key=cfg.openai_api_key)

    @property
    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection — useful when re-ingesting."""
        try:
            self._client.delete_collection(cfg.collection_name)
        except Exception:
            pass  # didn't exist
        self._collection = self._client.get_or_create_collection(
            name=cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: Sequence[Chunk], batch_size: int = 100) -> None:
        """Embed and add chunks. Batches to keep API calls reasonable."""
        if not chunks:
            return

        for start in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
            batch = chunks[start : start + batch_size]
            embeddings = self._embed([c.text for c in batch])
            self._collection.add(
                ids=[f"{c.source}::{c.chunk_index}" for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[
                    {"source": c.source, "section": c.section, "chunk_index": c.chunk_index}
                    for c in batch
                ],
            )

    def query(self, question: str, k: int | None = None) -> list[Hit]:
        """Return the top-k most similar chunks."""
        k = k or cfg.top_k
        embedding = self._embed([question])[0]
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
        )

        hits: list[Hit] = []
        if not result["ids"] or not result["ids"][0]:
            return hits

        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        ids = result["ids"][0]
        for doc, meta, dist, _id in zip(docs, metas, dists, ids):
            hits.append(Hit(
                text=doc,
                source=str(meta.get("source", "")),
                section=str(meta.get("section", "")),
                score=float(dist),
                id=str(_id),
            ))
        return hits

    def all_chunks(self) -> list[Hit]:
        """Return every chunk in the collection.

        Used by hybrid retrieval to build a BM25 index over the same corpus
        that's in the vector store. Distance is meaningless here and set to 0.
        """
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        out: list[Hit] = []
        for _id, doc, meta in zip(ids, docs, metas):
            out.append(Hit(
                text=doc,
                source=str((meta or {}).get("source", "")),
                section=str((meta or {}).get("section", "")),
                score=0.0,
                id=str(_id),
            ))
        return out

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embeddings API."""
        response = self._openai.embeddings.create(
            model=cfg.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]
