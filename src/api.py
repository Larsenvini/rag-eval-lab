"""HTTP layer over the RAG pipeline.

Endpoints:
    GET  /healthz       — liveness + readiness check (used by Docker, CI, k8s in Week 5)
    GET  /              — small status page describing the API
    POST /ask           — body: {"question": "...", "top_k": 5}, returns answer + citations

Run locally:
    uvicorn src.api:app --reload --port 8000

Then:
    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"question": "How does a Pod differ from a container?"}'

Or open the auto-generated Swagger UI: http://localhost:8000/docs
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import assert_ready, cfg
from src.generator import Generator
from src.retriever import Retriever


API_VERSION = "0.2.0"

_STATIC_DIR = Path(__file__).parent / "static"


# ─── Singletons ──────────────────────────────────────────────────────────
# Built once at startup so we don't re-init the OpenAI/Chroma clients on
# every request. lifespan() guarantees teardown is clean.

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    assert_ready()
    state["retriever"] = Retriever()
    state["generator"] = Generator()
    chunks = state["retriever"].store.count
    if chunks == 0:
        # Don't crash — let /healthz report it. But warn loudly so devs notice.
        print("⚠️  Vector store is empty. Run `python -m scripts.ingest` first.")
    else:
        print(f"✓ API ready — {chunks} chunks indexed")
    yield
    state.clear()


app = FastAPI(
    title="rag-eval-lab",
    version=API_VERSION,
    description=(
        "RAG over Kubernetes documentation with strict grounding and citation. "
        "See /docs for interactive API exploration."
    ),
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Permissive CORS — fine for portfolio demo. Tighten before any real prod use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ─────────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural-language question about Kubernetes.",
        examples=["How does a Pod differ from a container?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description=f"How many chunks to retrieve. Defaults to config ({cfg.top_k}).",
    )


class Citation(BaseModel):
    source: str = Field(..., description="Path of the source markdown file.")
    section: str = Field(..., description="Heading the chunk was under.")
    score: float = Field(..., description="Cosine distance from the query (lower = closer).")
    text: str = Field(..., description="The retrieved chunk content.")


class AskResponse(BaseModel):
    answer: str
    contexts: list[Citation]
    latency_ms: int = Field(..., description="End-to-end time on the server.")
    config: dict[str, Any] = Field(
        ...,
        description="The effective config used to produce this answer (for reproducibility).",
    )


class HealthResponse(BaseModel):
    status: str
    ready: bool
    chunks_indexed: int
    api_version: str
    embedding_model: str
    generation_model: str


# ─── Endpoints ────────────────────────────────────────────────────────────


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness + readiness probe.

    Returns 200 even if the store is empty (so a container orchestrator
    doesn't kill us during startup), but `ready` flips to true once chunks
    are indexed.
    """
    retriever: Retriever | None = state.get("retriever")
    chunks = retriever.store.count if retriever else 0
    return HealthResponse(
        status="ok",
        ready=chunks > 0,
        chunks_indexed=chunks,
        api_version=API_VERSION,
        embedding_model=cfg.embedding_model,
        generation_model=cfg.generation_model,
    )


@app.get("/", include_in_schema=False)
async def root():
    """Serve the chat UI."""
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")
    return FileResponse(index)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    retriever: Retriever = state["retriever"]
    generator: Generator = state["generator"]

    if retriever.store.count == 0:
        raise HTTPException(
            status_code=503,
            detail="Vector store is empty. Run `python -m scripts.ingest` first.",
        )

    t0 = time.perf_counter()
    contexts = retriever.retrieve(req.question, k=req.top_k)
    answer = generator.generate(req.question, contexts)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return AskResponse(
        answer=answer.text,
        contexts=[
            Citation(
                source=c.source,
                section=c.section,
                score=c.score,
                text=c.text,
            )
            for c in answer.contexts
        ],
        latency_ms=latency_ms,
        config={
            "embedding_model": cfg.embedding_model,
            "generation_model": cfg.generation_model,
            "top_k": req.top_k or cfg.top_k,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "temperature": cfg.temperature,
        },
    )
