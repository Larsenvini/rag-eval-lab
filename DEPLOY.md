# Deploying to Fly.io

This guide walks through a clean first-time deploy.

## What gets deployed

A Docker image containing:
- The FastAPI service (`src/api.py`) on port 8080
- The Kubernetes docs corpus, **already embedded** into ChromaDB at build time
- All Python deps

Runtime cost: one shared-CPU machine, 512MB RAM. Sleeps when idle, wakes on request (~5-10s cold start).

## Prereqs

```bash
# Install the Fly CLI: https://fly.io/docs/hands-on/install-flyctl/
# macOS:     brew install flyctl
# Windows:   iwr https://fly.io/install.ps1 -useb | iex
# Linux:     curl -L https://fly.io/install.sh | sh

flyctl auth signup   # or `flyctl auth login` if you have an account
```

## One-time setup

```bash
# From the repo root
flyctl launch --no-deploy

# When prompted:
# - "App name" → press Enter to use rag-eval-lab, or pick another if taken
# - "Region" → choose `gru` (São Paulo) for low latency from Brazil
# - "Postgres / Redis?" → No
# - "Deploy now?" → No (we set secrets first)
```

This creates the app on Fly's side and overwrites your `fly.toml` with their detected settings.
**Re-apply your local `fly.toml`** — `git checkout fly.toml` — because the one in this repo is already
tuned for our needs.

## Set the OpenAI key as a Fly secret

The key is needed at **build time** (to embed the corpus) AND at **runtime** (for the chat completions).

```bash
# Runtime secret (read by the running container)
flyctl secrets set OPENAI_API_KEY="sk-..."
```

## Deploy

The build also needs the key (for embedding). Pass it as a build secret:

```bash
flyctl deploy --build-secret OPENAI_API_KEY="$env:OPENAI_API_KEY"      # PowerShell
# OR
flyctl deploy --build-secret OPENAI_API_KEY=$OPENAI_API_KEY            # bash
```

First deploy takes **~5-8 minutes** (downloading the corpus, embedding ~500 chunks, building both image stages). Subsequent deploys are faster thanks to Docker layer caching.

## Verify it's live

```bash
# Health check
curl https://<your-app>.fly.dev/healthz

# Ask something
curl -X POST https://<your-app>.fly.dev/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "How does a Pod differ from a container?"}'

# Or just open the Swagger UI
# https://<your-app>.fly.dev/docs
```

## Updating after code changes

```bash
git push                                                       # commit your changes
flyctl deploy --build-secret OPENAI_API_KEY="$env:OPENAI_API_KEY"
```

The embeddings get regenerated every deploy. If you want to avoid that cost, see
"persistent volumes" in the Fly docs — but for portfolio scale, ~$0.05 per deploy is fine.

## Troubleshooting

```bash
flyctl logs              # tail runtime logs
flyctl status            # see machine state
flyctl ssh console       # shell into the running container
flyctl scale memory 1024 # bump memory if you see OOM
```

If `flyctl deploy` fails during the embed step (Stage 1), check:
1. Build secret is being passed correctly (`flyctl deploy --build-secret OPENAI_API_KEY=...`)
2. Your OpenAI account has credits / not rate-limited
3. The build log will show whether `download_corpus.py` or `ingest.py` failed

## Cost expectation

- Fly free tier covers small apps; expect $0–5/month for a low-traffic demo
- One embedding cycle (per deploy) ~$0.05
- Each `/ask` call to your live URL ~$0.001–0.005 depending on retrieval depth
- Set spending limits on both Fly and OpenAI before going public, just in case
