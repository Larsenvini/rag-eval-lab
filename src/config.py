"""Central configuration. Everything tunable lives here so we can A/B test in Week 3."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # API
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")

    # Models (cheap defaults — bump in Week 3 if needed)
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    generation_model: str = os.environ.get("GENERATION_MODEL", "gpt-4o-mini")

    # Storage
    chroma_path: Path = Path(os.environ.get("CHROMA_PATH", "./data/processed/chroma"))
    collection_name: str = os.environ.get("COLLECTION_NAME", "k8s_docs")
    raw_corpus_path: Path = Path(os.environ.get("RAW_CORPUS_PATH", "./data/raw"))

    # Chunking — Week 3 we'll sweep these
    chunk_size: int = int(os.environ.get("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", "120"))

    # Retrieval
    top_k: int = int(os.environ.get("TOP_K", "12"))
    default_retriever: str = os.environ.get("DEFAULT_RETRIEVER", "hybrid")

    # Generation
    temperature: float = float(os.environ.get("TEMPERATURE", "0.1"))
    max_answer_tokens: int = int(os.environ.get("MAX_ANSWER_TOKENS", "500"))


cfg = Config()


def assert_ready() -> None:
    """Fail fast with a useful message if the user hasn't configured their key."""
    if not cfg.openai_api_key or not cfg.openai_api_key.startswith("sk-"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing or malformed. "
            "Copy .env.example to .env and paste your key."
        )
