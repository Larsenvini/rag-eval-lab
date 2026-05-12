"""Tests for hybrid retriever — verify BM25 + RRF math without hitting OpenAI."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.hybrid_retriever import (
    HybridConfig,
    HybridRetriever,
    _tokenize,
)
from src.store import Hit


# ─── Tokenizer ────────────────────────────────────────────────────────────

def test_tokenizer_lowercases_and_splits():
    assert _tokenize("Hello World") == ["hello", "world"]


def test_tokenizer_keeps_hyphenated_terms():
    """k8s, config-map, node-port should survive as single tokens."""
    tokens = _tokenize("ConfigMap and node-port and k8s")
    assert "config-map" not in tokens  # 'and' is dropped, but...
    # 'configmap' is one word, 'node-port' is hyphenated
    assert "configmap" in tokens
    assert "node-port" in tokens
    assert "k8s" in tokens


def test_tokenizer_drops_stopwords():
    tokens = _tokenize("What is a Pod and how does it work")
    for sw in ["what", "is", "a", "and", "how", "does", "it"]:
        assert sw not in tokens
    assert "pod" in tokens
    assert "work" in tokens


def test_tokenizer_drops_pure_punctuation():
    tokens = _tokenize("Pods!!! ??? --- run")
    assert tokens == ["pods", "run"]


# ─── HybridRetriever ──────────────────────────────────────────────────────

def _fake_store_with_chunks(chunks: list[Hit]) -> MagicMock:
    """Build a mock VectorStore that returns our test chunks."""
    store = MagicMock()
    store.all_chunks.return_value = chunks
    # Default vector query returns nothing; tests override per-case
    store.query.return_value = []
    return store


def _chunks() -> list[Hit]:
    """A small toy corpus that lets us reason about retrieval by hand."""
    return [
        Hit(id="doc1", text="A Pod is the smallest deployable unit in Kubernetes.",
            source="pods.md", section="Intro", score=0.0),
        Hit(id="doc2", text="StatefulSets are used for stateful applications like databases.",
            source="statefulset.md", section="Intro", score=0.0),
        Hit(id="doc3", text="Deployments manage stateless frontend and backend applications.",
            source="deployment.md", section="Intro", score=0.0),
        Hit(id="doc4", text="A Service exposes a Pod or set of Pods to the network.",
            source="service.md", section="Intro", score=0.0),
        Hit(id="doc5", text="Connecting a frontend application to a backend API in Kubernetes.",
            source="connecting-frontend-backend.md", section="Intro", score=0.0),
    ]


def test_bm25_finds_keyword_match():
    """A query with 'frontend backend' should rank docs containing both words at the top.

    NOTE: doc3 ('frontend and backend applications') and doc5 ('frontend application
    to a backend') both contain both query terms. BM25 will tie them on score and
    fall back to insertion order. This is honest BM25 behavior — the win we care
    about is that BM25 surfaces these two over docs that lack the keywords entirely.
    """
    store = _fake_store_with_chunks(_chunks())
    hr = HybridRetriever(store=store)

    results = hr._bm25_search("deploying a frontend and backend", n=5)
    assert len(results) >= 2
    top_two_ids = {results[0].id, results[1].id}
    # Both keyword-rich docs should be in the top-2
    assert top_two_ids == {"doc3", "doc5"}, (
        f"Expected doc3 + doc5 (both contain 'frontend' AND 'backend'), got {top_two_ids}"
    )
    # And the docs that don't contain those words should NOT be there
    assert "doc1" not in top_two_ids
    assert "doc2" not in top_two_ids
    assert "doc4" not in top_two_ids


def test_bm25_returns_empty_for_no_match():
    """A query whose tokens never appear in the corpus returns empty."""
    store = _fake_store_with_chunks(_chunks())
    hr = HybridRetriever(store=store)
    results = hr._bm25_search("xylophone underwater photosynthesis", n=5)
    assert results == []


def test_rrf_fuses_two_lists():
    """RRF score for an item in both lists at rank 1 should be 2/(60+1) = ~0.033."""
    store = _fake_store_with_chunks(_chunks())
    hr = HybridRetriever(store=store, config=HybridConfig(rrf_k=60))

    list_a = [_chunks()[0], _chunks()[1]]   # doc1, doc2
    list_b = [_chunks()[1], _chunks()[2]]   # doc2, doc3

    fused = hr._rrf_fuse([list_a, list_b])
    fused_by_id = {h.id: h for h in fused}

    # doc2 appears at rank 2 in list_a (1/62) and rank 1 in list_b (1/61)
    expected_doc2 = 1 / (60 + 2) + 1 / (60 + 1)
    assert fused_by_id["doc2"].score == pytest.approx(expected_doc2, abs=1e-9)

    # doc1 appears only in list_a at rank 1 → 1/61
    expected_doc1 = 1 / (60 + 1)
    assert fused_by_id["doc1"].score == pytest.approx(expected_doc1, abs=1e-9)

    # doc2 should rank above doc1 (in both lists vs only one)
    assert fused[0].id == "doc2"


def test_rrf_handles_empty_lists():
    """A retriever returning nothing must not break fusion."""
    store = _fake_store_with_chunks(_chunks())
    hr = HybridRetriever(store=store)
    fused = hr._rrf_fuse([[], []])
    assert fused == []


def test_rrf_preserves_metadata_from_first_seen():
    """RRF should hand back text/source/section, not just an id."""
    store = _fake_store_with_chunks(_chunks())
    hr = HybridRetriever(store=store)
    fused = hr._rrf_fuse([[_chunks()[0]], [_chunks()[0]]])
    assert fused[0].source == "pods.md"
    assert "Pod" in fused[0].text


def test_retrieve_combines_bm25_and_vector():
    """End-to-end: hybrid retrieve uses both signals and returns top-k."""
    chunks = _chunks()
    store = _fake_store_with_chunks(chunks)

    # Pretend the vector store returns doc1 + doc4 for any query
    store.query.return_value = [chunks[0], chunks[3]]

    hr = HybridRetriever(
        store=store,
        config=HybridConfig(rrf_k=60, candidate_pool=10, final_k=3),
    )

    results = hr.retrieve("frontend backend application")

    # Should be at most final_k=3 results
    assert len(results) <= 3

    # doc5 (BM25 match for 'frontend backend') should appear in results
    ids = [r.id for r in results]
    assert "doc5" in ids


def test_retrieve_respects_k_override():
    chunks = _chunks()
    store = _fake_store_with_chunks(chunks)
    store.query.return_value = chunks[:3]
    hr = HybridRetriever(store=store, config=HybridConfig(candidate_pool=10))
    out = hr.retrieve("frontend", k=2)
    assert len(out) == 2


def test_empty_corpus_raises():
    """Building over an empty store should fail loudly, not silently."""
    store = MagicMock()
    store.all_chunks.return_value = []
    with pytest.raises(RuntimeError, match="empty"):
        HybridRetriever(store=store)


# ─── Hybrid + reranker composition ───────────────────────────────────────


def test_hybrid_with_reranker_uses_reranker():
    """When a reranker is provided, retrieve() routes through it."""
    chunks = _chunks()
    store = _fake_store_with_chunks(chunks)
    store.query.return_value = chunks[:3]

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [chunks[2]]   # whatever the reranker says

    hr = HybridRetriever(
        store=store,
        config=HybridConfig(candidate_pool=10, rerank_pool=10, final_k=3),
        reranker=fake_reranker,
    )

    out = hr.retrieve("any query")
    assert len(out) == 1
    assert out[0].id == "doc3"
    fake_reranker.rerank.assert_called_once()
    args, kwargs = fake_reranker.rerank.call_args
    # rerank was passed (query, pool, top_k=final_k)
    assert args[0] == "any query"
    assert kwargs.get("top_k") == 3


def test_hybrid_without_reranker_skips_rerank():
    chunks = _chunks()
    store = _fake_store_with_chunks(chunks)
    store.query.return_value = chunks[:3]

    hr = HybridRetriever(
        store=store,
        config=HybridConfig(candidate_pool=10, final_k=3),
        reranker=None,
    )
    out = hr.retrieve("any query")
    # Should just be RRF output, top 3
    assert len(out) <= 3
