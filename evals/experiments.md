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
