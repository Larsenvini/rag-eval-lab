# rag-eval-lab

> A production-shaped RAG system + automated evaluation framework, with CI gates that fail the build when answer quality drops.

**Status:** 🚧 Week 1 — Foundation
**Stack:** Python · OpenAI · ChromaDB · FastAPI · RAGAS · pytest · GitHub Actions · Docker
**Corpus:** Kubernetes documentation (Concepts + selected Tasks)

---

## Why this exists

Most "AI portfolio projects" stop at "I built a chatbot." This one stops at "I built a chatbot, then I tested it like software."

Coming from a QA Automation background, I think evaluation is the most underrated piece of LLM engineering. This project is my proof.

## What it does

1. **Ingests** the K8s Concepts + Tasks docs
2. **Chunks + embeds** them with `text-embedding-3-small`
3. **Stores** vectors in ChromaDB (local)
4. **Answers** questions over them with `gpt-4o-mini`, returning answer + citations
5. **Evaluates** answer quality on a hand-curated golden set using RAGAS
6. **Gates CI** — if faithfulness or relevance drops below threshold, the build fails

## Roadmap

- [ ] Week 1 — Ingestion + basic RAG (`python -m src.ask "..."` works)
- [ ] Week 2 — FastAPI endpoint + 30–50 question golden eval set
- [ ] Week 3 — Retrieval improvements (rerank, query expansion, chunk-size sweep)
- [ ] Week 4 — RAGAS in pytest + GitHub Actions quality gate
- [ ] Week 5 — Dockerize + deploy live URL
- [ ] Week 6 — Two writeups + portfolio integration

---

## Setup

```bash
# 1. Clone
git clone <repo-url> && cd rag-eval-lab

# 2. Create virtualenv
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate     # Windows

# 3. Install deps
pip install -r requirements.txt

# 4. Set your OpenAI key
cp .env.example .env
# edit .env, paste your sk-... key

# 5. Ingest the corpus (one-time, ~$0.05 in embeddings)
python -m scripts.ingest

# 6. Ask a question
python -m src.ask "How do I expose a deployment?"
```

## Project layout

```
rag-eval-lab/
├── src/                  # core RAG code
│   ├── config.py         # all knobs in one place
│   ├── chunker.py        # markdown → chunks
│   ├── store.py          # ChromaDB wrapper
│   ├── retriever.py      # query → relevant chunks
│   ├── generator.py      # chunks + question → answer
│   └── ask.py            # CLI entrypoint
├── scripts/
│   ├── download_corpus.py  # pull K8s docs from github
│   └── ingest.py           # chunk + embed + store
├── data/
│   ├── raw/              # markdown files as downloaded
│   └── processed/        # ChromaDB persistent store (gitignored)
├── evals/                # eval set + harness (Week 2+)
├── tests/                # pytest tests, incl. eval gate (Week 4)
└── .github/workflows/    # CI (Week 4)
```

## Cost estimate

For development across 6 weeks with frequent re-running:
- Embeddings (one-time): **~$0.05**
- Eval runs (~50 q/run × ~30 runs): **~$5**
- Manual queries while developing: **~$2**
- **Total: ~$10** (your budget is fine)

## License

MIT
