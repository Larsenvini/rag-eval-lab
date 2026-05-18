"""Smoke tests against the live deployed URL.

These are SYNTHETIC MONITORING tests, not unit tests. They hit the actual
deployed instance and verify it's serving requests correctly. They're
opt-in via the RUN_SMOKE environment variable so they never run as part
of the normal `pytest tests/` sweep — that suite stays fast and offline.

When to run these:
  - After a `flyctl deploy` to verify the new release works end-to-end
  - On a schedule (cron / GitHub Actions) to catch the live URL silently
    breaking due to corpus drift, API key expiry, dependency issues, etc.
  - Before a launch (LinkedIn post, recruiter share, etc.)

Usage:
  # Run against the default live URL
  RUN_SMOKE=1 pytest tests/test_smoke_live.py -v

  # Run against a custom URL (e.g. PR preview, staging)
  RUN_SMOKE=1 LIVE_URL=https://staging.fly.dev pytest tests/test_smoke_live.py -v

Why opt-in:
  - Cost: each smoke run hits OpenAI (one /ask call ≈ $0.003). Don't burn
    budget on every push.
  - Speed: cold-start wakeup can take 30-60s, which would slow normal CI.
  - Coupling: these tests fail if the *deploy* breaks, not just if the
    *code* broke. They shouldn't gate merging.
"""

from __future__ import annotations

import os
import re

import pytest

# Try to import requests; if unavailable, skip the whole module cleanly.
requests = pytest.importorskip("requests", reason="requests is needed for smoke tests")


LIVE_URL = os.environ.get("LIVE_URL", "https://rag-eval-lab.fly.dev").rstrip("/")
TIMEOUT = int(os.environ.get("SMOKE_TIMEOUT", "60"))


# Auto-skip the whole module if RUN_SMOKE isn't set. This keeps `pytest tests/`
# fast and prevents accidental cost during local development.
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SMOKE"),
    reason="Smoke tests are opt-in. Set RUN_SMOKE=1 to enable.",
)


# ─── Health ───────────────────────────────────────────────────────────────


def test_healthz_returns_200_and_ready():
    """The /healthz endpoint should return ready=True with the corpus loaded."""
    r = requests.get(f"{LIVE_URL}/healthz", timeout=TIMEOUT)
    assert r.status_code == 200, f"Health check returned {r.status_code}"

    body = r.json()
    assert body.get("status") == "ok"
    assert body.get("ready") is True, "Service reports it's not ready"
    assert body.get("chunks_indexed", 0) > 100, (
        f"Suspiciously few chunks indexed: {body.get('chunks_indexed')}. "
        "Did the corpus ingest fail?"
    )


def test_healthz_reports_expected_models():
    """The deployed instance should be using the locked production models."""
    r = requests.get(f"{LIVE_URL}/healthz", timeout=TIMEOUT)
    body = r.json()
    # We pin gpt-4o-mini as the generator and text-embedding-3-small as the embedder.
    # If these drift, something changed in config and we want to know.
    assert "gpt-4o" in (body.get("generation_model") or "")
    assert "embedding" in (body.get("embedding_model") or "")


# ─── /ask happy path ──────────────────────────────────────────────────────


def test_ask_returns_grounded_answer():
    """A basic question should get back a non-empty answer with retrieved sources."""
    r = requests.post(
        f"{LIVE_URL}/ask",
        json={"question": "What is a Pod in Kubernetes?"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200, f"/ask returned {r.status_code}: {r.text[:300]}"

    body = r.json()
    answer = body.get("answer", "")
    contexts = body.get("contexts", [])

    assert len(answer) >= 50, f"Answer suspiciously short ({len(answer)} chars): {answer!r}"
    assert len(contexts) >= 1, "No contexts returned — retrieval may be broken"
    # Sanity: a real K8s answer about Pods should mention the word "pod" somewhere
    assert "pod" in answer.lower(), "Answer doesn't mention 'pod' — likely off-topic"


def test_ask_returns_citations_in_text():
    """Answers should include [n]-style citations linking back to retrieved sources."""
    r = requests.post(
        f"{LIVE_URL}/ask",
        json={"question": "How do ConfigMaps differ from Secrets?"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200
    body = r.json()
    answer = body["answer"]

    # Look for at least one citation marker
    assert re.search(r"\[\d+(?:\s*,\s*\d+)*\]", answer), (
        f"No [n]-style citations found in answer. The generator prompt may have drifted.\n"
        f"Answer was: {answer[:500]}"
    )


def test_ask_returns_context_metadata():
    """Each retrieved context should have source path + text content."""
    r = requests.post(
        f"{LIVE_URL}/ask",
        json={"question": "What is a Deployment?"},
        timeout=TIMEOUT,
    )
    body = r.json()
    contexts = body.get("contexts", [])
    assert contexts, "No contexts to inspect"

    first = contexts[0]
    assert first.get("source"), "Context missing 'source' field"
    assert first.get("text"), "Context missing 'text' field"
    assert len(first["text"]) > 20, "Context text suspiciously short"


# ─── /ask validation ──────────────────────────────────────────────────────


def test_ask_rejects_empty_question():
    """The API should reject empty/short questions with 4xx, not crash."""
    r = requests.post(
        f"{LIVE_URL}/ask",
        json={"question": ""},
        timeout=TIMEOUT,
    )
    # 422 (FastAPI validation) or 400 are both acceptable. Anything 5xx is a bug.
    assert 400 <= r.status_code < 500, (
        f"Expected client error for empty question, got {r.status_code}"
    )


def test_ask_rejects_missing_field():
    """The API should reject malformed payloads with 422."""
    r = requests.post(
        f"{LIVE_URL}/ask",
        json={},                          # missing 'question'
        timeout=TIMEOUT,
    )
    assert r.status_code == 422


# ─── UI ───────────────────────────────────────────────────────────────────


def test_ui_loads_at_root():
    """GET / should return the chat UI as HTML, not the JSON health response."""
    r = requests.get(LIVE_URL, timeout=TIMEOUT)
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", ""), (
        f"Root returned {r.headers.get('content-type')}, expected HTML"
    )
    text = r.text
    assert "<title>Rag Eval Lab by Larsenvini</title>" in text, "UI title missing — wrong page served?"
    assert "Kubernetes" in text, "UI is missing expected Kubernetes branding"


def test_ui_references_api_endpoints():
    """The UI should reference the /ask and /healthz endpoints it depends on."""
    r = requests.get(LIVE_URL, timeout=TIMEOUT)
    text = r.text
    assert "/ask" in text, "UI doesn't reference /ask endpoint"
    assert "/healthz" in text, "UI doesn't reference /healthz endpoint"


# ─── Swagger docs ─────────────────────────────────────────────────────────


def test_swagger_docs_available():
    """The /docs page should still be accessible for engineers."""
    r = requests.get(f"{LIVE_URL}/docs", timeout=TIMEOUT)
    assert r.status_code == 200
    # FastAPI's Swagger UI page contains this signature
    assert "swagger-ui" in r.text.lower()
