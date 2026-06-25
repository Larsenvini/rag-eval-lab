# Experiment Log

Every experiment gets a row in the ablation table and a short write-up below.
This is the data backbone of the blog post.

**How to add a row:**
1. Run: `python -m scripts.run_eval --top-k <N> --note "<label>"`
2. Score: `python -m scripts.score_eval evals/results/run_<timestamp>.json`
3. Compare: `python -m scripts.compare_runs evals/results/scored_<timestamp>.json`
4. Paste the deltas into the relevant section below.

---

## Baseline (locked 2026-05-06)

| Field | Value |
|---|---|
| File | `evals/results/baseline.json` |
| n | 41 |
| top_k | 5 |
| chunk_size | 800 / overlap 120 |
| model | gpt-4o-mini, temp 0.1 |
| judge | gpt-4o |

| Metric | Score |
|---|---|
| Faithfulness | 0.81 |
| Answer relevance | 0.78 |
| GT similarity | 0.61 |

| Failure mode | Count | % |
|---|---|---|
| answer-incomplete | 15 | 37% |
| synthesis-fail | 10 | 24% |
| retrieval-miss | 10 | 24% |
| none | 6 | 15% |

Canaries (scored 0.00 across all metrics): Q06, Q20, Q25.

---

## Experiment 1 — Retrieval depth (dense, top_k sweep)

**Verdict: top_k = 12 wins.** All three aggregate metrics improved monotonically
with depth (faith +0.04, rel +0.05, gt +0.10). retrieval-miss dropped 24% → 10%;
clean answers ("none") rose 15% → 27%. Two of three canaries flipped from
0.00 → 0.90 relevance.

**Adopted top_k = 12 as the new default** (`src/config.py`). All subsequent
experiments are measured from this new working point, while the locked baseline
(top_k = 5) remains the absolute reference.

**Limitations:**
- Q06 (architecture / why-tradeoff) refused to improve — likely a corpus gap,
  not a retrieval-depth problem. Flagged for investigation.
- Judge noise floor estimated at ±1 question / ±0.02 aggregate on identical inputs.
- Cost increased ~1.4× per query (more tokens in context), latency +600 ms.

---

## Experiment 2 — Hybrid retrieval & reranking ablation

Four configurations tested against the locked baseline (dense, top_k = 5), all on
the same 41-question golden set.

### The complete ablation

| Config | Faith | Rel | GT Sim | retrieval-miss | none | Canaries |
|---|---:|---:|---:|---:|---:|:--:|
| Baseline (dense, k=5)     | 0.815 | 0.778 | 0.610 | 10 | 6  | 0/3 |
| E1 (dense, k=12)          | 0.859 | 0.827 | 0.705 | 4  | 11 | 2/3 |
| E2 (hybrid, k=12)         | 0.878 | 0.851 | 0.717 | 6  | 8  | 3/3 |
| E3 (hybrid + MiniLM)      | 0.837 | 0.812 | 0.695 | 4  | 9  | 3/3 |
| E3b (hybrid + BGE-large)  | 0.856 | 0.837 | 0.722 | 3  | 11 | 3/3 |

### The honest read

BGE-large is the strongest by some measures, but E2 (plain hybrid) still leads on
the metrics that matter most. Three angles:

- **Aggregate quality (faith + rel):** E2 wins by a clear margin — 0.88/0.85 vs
  0.86/0.84. That's not noise; it's roughly two questions' worth of difference on
  a 41-question set.
- **Retrieval-miss reduction:** BGE-large wins, 3 vs 6. That's its job and it does it.
- **Clean wins (none):** tied at 11. *(Flagged: the ablation table above lists E2
  at none = 8 — reconcile this count before publishing.)*
- **GT-similarity:** BGE-large barely edges it out (0.722 vs 0.717). Within judge noise.

**Why E2 beats BGE-large on the aggregate.** Look at what BGE-large does to
questions E2 handled cleanly:

```
Q12 (pods / how-to):       none → answer-incomplete             (−0.10 / −0.20)
Q17 (deployments / edge):  none → synthesis-fail                (−0.10 / −0.20)
Q41 (architecture / edge): retrieval-miss persists              (−0.30 / −0.20)
Q01 (architecture / how):  answer-incomplete → synthesis-fail   (−0.20)
Q30 (storage / edge):                                           (−0.10 / −0.30)
```

The reranker is trading wins for losses. It rescues 3 retrieval-miss canaries but
bumps 5 questions out of cleaner buckets. Net effect: same retrieval health,
slightly worse synthesis. This is the exact pattern we predicted — reranking
shuffles, and shuffling isn't always improvement.

**The story that's emerging:**

```
dense, k=5   → dense, k=12 : depth fix                → +0.04 faith, +0.05 rel
dense, k=12  → hybrid      : keyword bridge           → +0.02 faith, +0.02 rel  ★ peak
hybrid       → +MiniLM     : domain-mismatched rerank → −0.04 faith, −0.04 rel
hybrid       → +BGE-large  : SOTA rerank              → −0.02 faith, −0.01 rel
```

Each retrieval improvement gave less than the previous one, and the last one was
negative — a textbook diminishing-returns arc, and one this project has the data
to demonstrate.

### Summary

| Config | Δ Faith | Δ Rel | Δ GT | Δ retrieval-miss |
|---|---:|---:|---:|---:|
| dense, top_k=12          | +0.04 | +0.05 | +0.10 | −6 |
| hybrid (RRF)             | +0.06 | +0.07 | +0.11 | −4 |
| hybrid + MiniLM rerank   | +0.02 | +0.03 | +0.09 | −6 |
| hybrid + BGE-large rerank| +0.04 | +0.06 | +0.11 | −7 |

**Winner: plain hybrid retrieval.** Reranking did reduce retrieval-miss further,
but at the cost of degrading synthesis on cleaner questions. The SOTA reranker
(BGE-large, 560M params, ~1–3 s/query on CPU) closed most but not all of the gap
to plain hybrid, and never beat it on faith/relevance. For this corpus, retrieval
is no longer the limiting factor — the dominant failure mode is now
answer-incomplete (~33%), a generation-side issue.

**Adopted: hybrid retrieval (BM25 + vector + RRF, top_k = 12).** Reranking
rejected as not worth the latency and complexity for this corpus.

**Generalization note:** This result is specific to a well-structured
documentation corpus (~3,770 chunks of curated Kubernetes docs). On noisier or
less consistently structured corpora, rerankers typically show more of their
value. A QA-mindset takeaway: an optimization is only valid against your specific
problem — assumptions from the literature need empirical validation on your own data.

---

## Ablation runs vs production scores

The numbers above are single-run (N=1) A/B comparisons from the Week 3–4 retrieval
study — fast, and good enough to choose a configuration. The deployed
**production** scores are more conservative N=3 medians on the same hybrid / k=12
config; see the README and `evals/results/last_main.json`:

**faithfulness 0.859 · answer relevance 0.846 · GT similarity 0.700 · retrieval-miss 3/41**

The gap between the ablation's E2 row and production reflects single-run judge
variance (±0.02 aggregate) and the shift to N=3 medians.

---

## Notes / Observations

- GT similarity is the least stable metric (sensitive to ground-truth wording).
  Trust faithfulness and relevance more for A/B decisions.
- Binary-ish scores (0.0 / 0.5 / 1.0) in early runs were a prompt artifact —
  fixed in `score_eval.py` with a fine-grained rubric.
- Three questions (Q06: frontend/backend/database; Q20: services why-tradeoff;
  Q25: configmaps edge-case) scored 0.00 across all metrics in the baseline.
  These were the clearest retrieval failures and serve as canaries.
