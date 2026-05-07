# rag-eval-lab

> A production-shaped RAG system + automated evaluation framework, with CI gates that fail the build when answer quality drops.

**Status:** Week 2 complete · Baseline locked
**Stack:** Python · OpenAI · ChromaDB · FastAPI · pytest · GitHub Actions (Week 4) · Docker (Week 5)
**Corpus:** Kubernetes documentation (Concepts + selected Tasks)

---

## Why this exists

Most "AI portfolio projects" stop at "I built a chatbot." This one stops at "I built a chatbot, then I tested it like software."

Coming from a QA Automation background, I think evaluation is the most underrated piece of LLM engineering. This project is my proof.

## What it does

1. **Ingests** the K8s Concepts + Tasks docs
2. **Chunks + embeds** them with `text-embedding-3-small`
3. **Stores** vectors in ChromaDB (local, persistent)
4. **Answers** questions over them with `gpt-4o-mini`, returning answer + citations
5. **Serves** the pipeline behind a FastAPI endpoint with health checks
6. **Evaluates** answer quality on a curated golden set using LLM-as-judge
7. **Will gate CI** (Week 4) — if faithfulness or relevance drops below threshold, the build fails

## Roadmap

- [x] **Week 1** — Ingestion + basic RAG (CLI: `python -m src.ask "..."`)
- [x] **Week 2** — FastAPI endpoint + 41-question golden eval set + baseline numbers
- [ ] **Week 3** — Retrieval improvements (rerank, query expansion, chunk-size sweep) measured against baseline
- [ ] **Week 4** — Pytest eval suite + GitHub Actions quality gate
- [ ] **Week 5** — Dockerize + deploy live URL
- [ ] **Week 6** — Two writeups + portfolio integration

---

## Baseline (Week 2)

Locked on **2026-05-06**. Every Week 3 experiment is measured against `evals/results/baseline.json`.

**Configuration:**
- Generation: `gpt-4o-mini`, temperature 0.1, top_k 5
- Embeddings: `text-embedding-3-small`
- Chunking: chunk_size 800, overlap 120
- Eval set: 41 questions (11 manually written + 30 LLM-drafted, human-verified)
- Judge: `gpt-4o`

**Scores:**

| Metric                  | Score |
|-------------------------|-------|
| Faithfulness            | 0.81  |
| Answer relevance        | 0.78  |
| Ground-truth similarity | 0.61  |

**Failure mode breakdown:**

| Mode                | Count | %   |
|---------------------|-------|-----|
| answer-incomplete   | 15    | 37% |
| synthesis-fail      | 10    | 24% |
| retrieval-miss      | 10    | 24% |
| none (clean wins)   |  6    | 15% |

**Diagnosis:** The model is grounded (faithfulness 0.81) and on-topic (relevance 0.78). The dominant problem is **depth, not relevance** — answers are systematically shallower than the ground-truth references. Q06, Q20, and Q25 score 0.00 across all metrics and serve as canaries for retrieval improvements.

---

## Setup

```bash
# 1. Clone and enter
git clone <repo-url> && cd rag-eval-lab

# 2. Create virtualenv
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# 3. Install
pip install -r requirements.txt

# 4. Set your OpenAI key
cp .env.example .env
# edit .env, paste sk-...

# 5. Ingest the corpus (one-time, ~$0.05 in embeddings)
python -m scripts.download_corpus
python -m scripts.ingest

# 6. Use it — pick one
python -m src.ask "How does a Pod differ from a container?"
uvicorn src.api:app --reload --port 8000
```

## Using the API

Once `uvicorn src.api:app --reload` is running:

```bash
# Liveness check
curl http://localhost:8000/healthz

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I expose a service externally?"}'
```

Or open Swagger UI: http://localhost:8000/docs

## Running the eval pipeline

```bash
# Run all golden questions through the RAG, save raw results
python -m scripts.run_eval

# Score the latest run with LLM-as-judge
python -m scripts.score_eval

# Or do both in one go
python -m scripts.eval_pipeline
```

## Running tests

```bash
pytest tests/ -q
```

These are offline tests (no OpenAI calls) — they run in CI for free.

## Project layout

```
rag-eval-lab/
├── src/                    # core RAG code
│   ├── config.py           # all knobs in one place
│   ├── chunker.py          # markdown → chunks
│   ├── store.py            # ChromaDB wrapper
│   ├── retriever.py        # query → relevant chunks
│   ├── generator.py        # chunks + question → answer
│   ├── ask.py              # CLI entrypoint
│   └── api.py              # FastAPI service
├── scripts/
│   ├── download_corpus.py  # pull K8s docs from github
│   ├── ingest.py           # chunk + embed + store
│   ├── generate_questions.py  # coverage matrix + GPT-4o question generator
│   ├── run_eval.py         # run golden set through RAG
│   ├── score_eval.py       # LLM-as-judge scoring
│   └── eval_pipeline.py    # orchestrates the three steps
├── data/
│   ├── raw/                # markdown files as downloaded (gitignored)
│   └── processed/          # ChromaDB persistent store (gitignored)
├── evals/
│   ├── golden_set.json     # the 41-question evaluation set
│   ├── taxonomy.py         # canonical failure-mode enum
│   └── results/
│       ├── baseline.json   # immutable Week 2 baseline
│       └── *.json          # per-experiment results (gitignored)
├── tests/                  # pytest tests, incl. eval gate (Week 4)
└── .github/workflows/      # CI (Week 4)
```

## Cost estimate

For 6 weeks of development with frequent re-running:
- Embeddings (one-time): ~$0.05
- Eval runs (~30 runs × $0.50): ~$15
- Manual queries while developing: ~$2
- **Total: ~$17**

## License

MIT
