"""Tests for the cross-encoder reranker.

We mock the underlying CrossEncoder so tests are fast and don't download
~80MB of model weights in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.reranker import Reranker, RerankConfig
from src.store import Hit


def _hits() -> list[Hit]:
    return [
        Hit(id="a", text="apple pie", source="a.md", section="Intro", score=0.0),
        Hit(id="b", text="banana split", source="b.md", section="Intro", score=0.0),
        Hit(id="c", text="cherry cobbler", source="c.md", section="Intro", score=0.0),
    ]


@pytest.fixture
def fake_model():
    """A stand-in for sentence_transformers.CrossEncoder.

    .predict() returns whatever scores you set on .next_scores.
    """
    m = MagicMock()
    m.next_scores = []
    m.predict.side_effect = lambda pairs: m.next_scores
    return m


def _patch_ce(fake_model):
    """Patch the import of CrossEncoder inside reranker._ensure_model."""
    return patch(
        "sentence_transformers.CrossEncoder",
        return_value=fake_model,
    )


def test_rerank_orders_by_predicted_score(fake_model):
    fake_model.next_scores = [0.1, 0.9, 0.5]  # b is highest, then c, then a

    with _patch_ce(fake_model):
        r = Reranker()
        out = r.rerank("dessert", _hits())

    assert [h.id for h in out] == ["b", "c", "a"]
    assert out[0].score == pytest.approx(0.9)
    assert out[1].score == pytest.approx(0.5)
    assert out[2].score == pytest.approx(0.1)


def test_rerank_preserves_metadata(fake_model):
    fake_model.next_scores = [0.5, 0.5, 0.5]

    with _patch_ce(fake_model):
        r = Reranker()
        out = r.rerank("dessert", _hits())

    # source/section/text/id all preserved across reranking
    sources = {h.source for h in out}
    assert sources == {"a.md", "b.md", "c.md"}


def test_rerank_respects_top_k(fake_model):
    fake_model.next_scores = [0.1, 0.9, 0.5]

    with _patch_ce(fake_model):
        r = Reranker()
        out = r.rerank("dessert", _hits(), top_k=2)

    assert len(out) == 2
    assert [h.id for h in out] == ["b", "c"]


def test_rerank_handles_empty_hits():
    """No model load should happen if there's nothing to score."""
    r = Reranker()
    assert r.rerank("anything", []) == []
    # Model should never have been loaded
    assert r._model is None


def test_model_is_lazy_loaded(fake_model):
    """Constructing a Reranker should NOT download the model.

    The model loads only on the first rerank call.
    """
    with _patch_ce(fake_model) as ce_class:
        r = Reranker()
        assert r._model is None
        ce_class.assert_not_called()  # not yet

        fake_model.next_scores = [0.5]
        r.rerank("q", [_hits()[0]])
        ce_class.assert_called_once()


def test_model_is_loaded_only_once(fake_model):
    """Successive rerank calls reuse the loaded model."""
    with _patch_ce(fake_model) as ce_class:
        r = Reranker()
        fake_model.next_scores = [0.5]
        r.rerank("q1", [_hits()[0]])
        fake_model.next_scores = [0.7]
        r.rerank("q2", [_hits()[1]])

        # CrossEncoder constructor was only called the first time
        assert ce_class.call_count == 1


def test_custom_model_name_passed_through(fake_model):
    with _patch_ce(fake_model) as ce_class:
        r = Reranker(RerankConfig(model_name="some/other-model"))
        fake_model.next_scores = [0.5]
        r.rerank("q", [_hits()[0]])
        ce_class.assert_called_once_with("some/other-model")
