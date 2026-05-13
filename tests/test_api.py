"""API smoke tests — no real OpenAI calls.

We patch the Retriever and Generator so these tests are free, fast, and
runnable in CI without secrets.

Run: pytest tests/test_api.py -q
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_state():
    """Build a fake retriever + generator that don't hit any APIs."""
    fake_retriever = MagicMock()
    fake_retriever.store.count = 100
    fake_hit = MagicMock(
        text="A Pod is the smallest deployable unit in Kubernetes.",
        source="concepts/workloads/pods.md",
        section="What is a Pod?",
        score=0.18,
    )
    fake_retriever.retrieve.return_value = [fake_hit]

    fake_answer = MagicMock(
        text="A Pod is the smallest deployable unit in Kubernetes. [1]",
        contexts=[fake_hit],
    )
    fake_generator = MagicMock()
    fake_generator.generate.return_value = fake_answer

    return {"retriever": fake_retriever, "generator": fake_generator}


@pytest.fixture
def client(mock_state):
    # Patch assert_ready so we don't need an OPENAI_API_KEY in CI
    with patch("src.api.assert_ready"), \
         patch("src.api.Retriever", return_value=mock_state["retriever"]), \
         patch("src.api.Generator", return_value=mock_state["generator"]):
        from src.api import app
        with TestClient(app) as c:
            yield c


def test_root_returns_metadata(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "rag-eval-lab"
    assert "version" in body


def test_healthz_reports_ready(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["chunks_indexed"] == 100


def test_ask_happy_path(client):
    r = client.post("/ask", json={"question": "What is a Pod?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert isinstance(body["citations"], list)
    assert len(body["citations"]) == 1
    assert body["citations"][0]["source"] == "concepts/workloads/pods.md"
    assert "latency_ms" in body
    assert "config" in body


def test_ask_rejects_short_question(client):
    r = client.post("/ask", json={"question": "Hi"})
    assert r.status_code == 422  # Pydantic validation


def test_ask_rejects_missing_field(client):
    r = client.post("/ask", json={})
    assert r.status_code == 422


def test_ask_returns_503_when_store_empty(mock_state):
    """If the vector store is empty, /ask must refuse with 503."""
    mock_state["retriever"].store.count = 0
    with patch("src.api.assert_ready"), \
         patch("src.api.Retriever", return_value=mock_state["retriever"]), \
         patch("src.api.Generator", return_value=mock_state["generator"]):
        from src.api import app
        with TestClient(app) as c:
            r = c.post("/ask", json={"question": "What is a Pod?"})
            assert r.status_code == 503
