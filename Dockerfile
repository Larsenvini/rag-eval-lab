# syntax=docker/dockerfile:1.7

# ─── Stage 1: BUILD ───────────────────────────────────────────────────────
# Install Python deps, clone the corpus, and run the embedding ingest.
# The build secret is the OpenAI key — we need it to embed but don't want
# it in the final image. Pass at build time:
#   fly deploy --build-secret OPENAI_API_KEY=$OPENAI_API_KEY
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# git is needed for download_corpus.py; build-essential for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python deps first for layer caching
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy source. NOTE: we do NOT copy data/ or evals/results/ — those are
# either regenerated here (data/processed) or are dev-only artifacts.
COPY src/         ./src/
COPY scripts/     ./scripts/

# Embed the corpus into a persistent ChromaDB on disk.
# The build secret is mounted into /run/secrets and read into env.
RUN --mount=type=secret,id=OPENAI_API_KEY \
    OPENAI_API_KEY="$(cat /run/secrets/OPENAI_API_KEY)" \
    python -m scripts.download_corpus && \
    OPENAI_API_KEY="$(cat /run/secrets/OPENAI_API_KEY)" \
    python -m scripts.ingest


# ─── Stage 2: RUNTIME ─────────────────────────────────────────────────────
# Slim image, only carries the embedded corpus + source + deps.
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Curl for healthcheck; no build tools needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app
WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source and the embedded corpus
COPY --from=builder /build/src     ./src
COPY --from=builder /build/scripts ./scripts
COPY --from=builder /build/data/processed ./data/processed

RUN chown -R app:app /app
USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8080/healthz || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
