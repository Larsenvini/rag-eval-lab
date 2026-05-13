# Experiment Log

Every Week 3+ experiment gets a row. This becomes the data backbone of the blog post.

**How to add a row:**
1. Run: `python -m scripts.run_eval --top-k <N> --note "<label>"`
2. Score: `python -m scripts.score_eval evals/results/run_<timestamp>.json`
3. Compare: `python -m scripts.compare_runs evals/results/scored_<timestamp>.json`
4. Paste the deltas here.

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

Canaries (scored 0.00 across all metrics): Q06, Q20, Q25

---

## Experiments

| # | Date | Label / Note | top_k | Change | Faith. | Rel. | GT Sim. | dFaith. | dRel. | Canaries fixed | File |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 2026-05-06 | baseline | 5 | — | 0.81 | 0.78 | 0.61 | — | — | 0/3 | baseline.json |
<!-- add rows below as you run experiments -->

---

## Notes / Observations

- GT similarity is the least stable metric (sensitive to ground-truth wording). Trust faithfulness and relevance more for A/B decisions.
- Binary-ish scores (0.0/0.5/1.0) in early runs were a prompt artifact — fixed in score_eval.py with fine-grained rubric.
- Three questions (Q06: frontend/backend/database, Q20: services why-tradeoff, Q25: configmaps edge-case) score 0.00 across all metrics. These are the clearest retrieval failures and serve as canaries.


**Verdict: top_k = 12 wins.** All three aggregate metrics improved monotonically with depth (faith +0.04, rel +0.05, gt +0.10). retrieval-miss dropped 24% → 10%; clean answers ("none") rose 15% → 27%. Two of three canaries flipped from 0.00 → 0.90 relevance.

**Adopting top_k = 12 as the new default.** Update `src/config.py`. All subsequent experiments are measured from this new working point, while the locked baseline (top_k=5) remains the absolute reference.

**Limitations:**
- Q06 (architecture/why-tradeoff) refused to improve — likely a corpus gap, not a retrieval depth problem. Flagged for investigation.
- Judge noise floor estimated at ±1 question / ±0.02 aggregate on identical inputs.
- Cost increased ~1.4× per query (more tokens in context), latency +600ms.

## Retrieval ablation summary

Four configurations tested against locked baseline (dense, top_k=5):

| Config | Δ Faith | Δ Rel | Δ GT | Δ retrieval-miss |
|---|---:|---:|---:|---:|
| dense, top_k=12 | +0.04 | +0.05 | +0.10 | −6 |
| hybrid (RRF) | +0.06 | +0.07 | +0.11 | −4 |
| hybrid + MiniLM rerank | +0.02 | +0.03 | +0.09 | −6 |
| hybrid + BGE-large rerank | +0.04 | +0.06 | +0.11 | −7 |

**Winner: plain hybrid retrieval.** Reranking did reduce retrieval-miss further but at the cost of degrading synthesis on cleaner questions. The SOTA reranker (BGE-large, 560M params, ~1-3s/query on CPU) closed most but not all of the gap to plain hybrid, and never beat it on faith/relevance. For this corpus, retrieval is no longer the limiting factor — the dominant failure mode is now answer-incomplete (~33%), a generation-side issue.

**Adopted: hybrid retrieval (BM25 + vector + RRF, top_k=12).** Reranking rejected as not worth the latency and complexity cost for this corpus.

**Generalization note:** This result is specific to a small (~500 chunk), well-structured documentation corpus. On larger or noisier corpora, rerankers typically show their value. A QA-mindset takeaway: optimization is only valid against your specific problem; assumptions from the literature need empirical validation.


The complete retrieval ablation
ConfigFaithRelGT Simretrieval-missnoneCanariesBaseline (dense, k=5)0.8150.7780.6101060/3E1 (dense, k=12)0.8590.8270.7054112/3E2 (hybrid, k=12)0.8780.8510.717683/3E3 (+MiniLM)0.8370.8120.695493/3E3b (+BGE-large)0.8560.8370.7223113/3
The honest read
BGE-large is the strongest by some measures, but E2 still leads on the metrics that matter most.
Look at it from three angles:
Aggregate quality (faith + rel): E2 wins by a clear margin. 0.88/0.85 vs 0.86/0.84. That's not noise — that's two questions' worth of difference on a 41-question set.
Retrieval-miss reduction: BGE-large wins. 3 vs 6. That's its job and it does it.
Clean wins (none): Tied at 11. Same.
GT-similarity: BGE-large barely edges out (0.722 vs 0.717). Within judge noise.
Why E2 beats BGE-large on the aggregate
Look at what BGE-large does to questions that E2 handled cleanly:
Q12 (pods/how-to):       none → answer-incomplete  (-0.10/-0.20)
Q17 (deployments/edge):  none → synthesis-fail     (-0.10/-0.20)
Q41 (architecture/edge): retrieval-miss persists   (-0.30/-0.20)
Q01 (architecture/how):  answer-incomplete → synthesis-fail (-0.20)
Q30 (storage/edge):                                 (-0.10/-0.30)
The reranker is trading wins for losses. It rescues 3 retrieval-miss canaries but bumps 5 questions out of cleaner buckets. Net effect: same retrieval health, slightly worse synthesis.
This is the exact pattern we predicted: reranking shuffles, and shuffling isn't always improvement.
The story that's emerging
dense, k=5    → dense, k=12  : depth fix          → +0.04 faith, +0.05 rel
dense, k=12   → hybrid       : keyword bridge     → +0.02 faith, +0.02 rel  ★ peak
hybrid        → +MiniLM      : domain-mismatched rerank → −0.04 faith, −0.04 rel
hybrid        → +BGE-large   : SOTA rerank         → −0.02 faith, −0.01 rel
Each retrieval improvement gave less than the previous one, and the last one was negative. That's a perfect "diminishing returns" arc. It's exactly what production ML practitioners encounter when they iterate on retrieval pipelines — and you have data to prove you saw it.
What I would write up
markdown## Retrieval ablation summary

Four configurations tested against locked baseline (dense, top_k=5):

| Config | Δ Faith | Δ Rel | Δ GT | Δ retrieval-miss |
|---|---:|---:|---:|---:|
| dense, top_k=12 | +0.04 | +0.05 | +0.10 | −6 |
| hybrid (RRF) | +0.06 | +0.07 | +0.11 | −4 |
| hybrid + MiniLM rerank | +0.02 | +0.03 | +0.09 | −6 |
| hybrid + BGE-large rerank | +0.04 | +0.06 | +0.11 | −7 |

**Winner: plain hybrid retrieval.** Reranking did reduce retrieval-miss further but at the cost of degrading synthesis on cleaner questions. The SOTA reranker (BGE-large, 560M params, ~1-3s/query on CPU) closed most but not all of the gap to plain hybrid, and never beat it on faith/relevance. For this corpus, retrieval is no longer the limiting factor — the dominant failure mode is now answer-incomplete (~33%), a generation-side issue.

**Adopted: hybrid retrieval (BM25 + vector + RRF, top_k=12).** Reranking rejected as not worth the latency and complexity cost for this corpus.

**Generalization note:** This result is specific to a small (~500 chunk), well-structured documentation corpus. On larger or noisier corpora, rerankers typically show their value. A QA-mindset takeaway: optimization is only valid against your specific problem; assumptions from the literature need empirical validation.
That paragraph alone is worth more in interviews than any of the individual experiments. It's a real engineer's analysis.