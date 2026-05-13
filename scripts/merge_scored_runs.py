"""Merge N scored eval runs into a single result with per-question
distributions and stability classifications.

Pure functions only — no I/O, no OpenAI calls. The wrapper script
(run_eval_n.py) handles reading/writing files; this module handles the math.

Why this split:
  Eval merging logic is the kind of thing you want unit-tested with
  hand-crafted inputs. Mixing it with file I/O makes that painful.

Stability classification:
  We use std-dev of `answer_relevance` across runs (the most narrative
  metric) to classify each question:
    stable        : std <= 0.10   — judgment is consistent
    mild-variance : 0.10 < std <= 0.25  — minor noise
    bimodal       : std > 0.25    — the judge is sometimes drastically
                                     different. Often signals a borderline
                                     refusal/synthesis case (e.g. Q06).

  These bands are calibrated to the 0-1 score scale our judge uses and
  match published guidance on LLM-as-judge variance (see e.g. RAGAS docs).
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any


METRICS = ["faithfulness", "answer_relevance", "ground_truth_similarity"]

# Stability thresholds, applied to std-dev of answer_relevance across runs.
STABLE_MAX_STD = 0.10
MILD_MAX_STD = 0.25


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def _std(xs: list[float]) -> float:
    # pstdev (population std) is fine here — we're summarizing a finite
    # sample, not inferring about a parent distribution.
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def classify_stability(rel_std: float) -> str:
    """Return 'stable' | 'mild-variance' | 'bimodal' based on relevance std."""
    if rel_std <= STABLE_MAX_STD:
        return "stable"
    if rel_std <= MILD_MAX_STD:
        return "mild-variance"
    return "bimodal"


def _summarize(samples: list[float]) -> dict[str, Any]:
    """Compute median/mean/std/samples for one metric across runs."""
    return {
        "median": round(_median(samples), 4),
        "mean": round(_mean(samples), 4),
        "std": round(_std(samples), 4),
        "samples": [round(s, 4) for s in samples],
    }


def merge_per_question(per_run_results: list[list[dict]]) -> list[dict]:
    """Given N runs (each a list of per-question result dicts), merge into one
    list of per-question dicts with score distributions and stability tags.

    Assumes all runs cover the same question ids. Questions present in
    fewer than all runs are still included, with the available samples.
    """
    if not per_run_results:
        return []

    # Index every run by question id
    indexed: list[dict[str, dict]] = [
        {r["id"]: r for r in run if "id" in r}
        for run in per_run_results
    ]
    all_ids = sorted({qid for run_index in indexed for qid in run_index.keys()})

    merged: list[dict] = []
    for qid in all_ids:
        # Collect samples for this question from every run that has it
        question_runs = [run_index[qid] for run_index in indexed if qid in run_index]
        if not question_runs:
            continue

        # First run is the source of question metadata (topic, type, ground_truth, etc.)
        # Static fields are the same across runs by construction.
        first = question_runs[0]
        scores = {
            m: _summarize([float(r.get(m, 0.0)) for r in question_runs])
            for m in METRICS
        }

        failure_modes = Counter(r.get("failure_mode", "unknown") for r in question_runs)
        rel_std = scores["answer_relevance"]["std"]

        merged.append({
            "id": qid,
            "topic": first.get("topic"),
            "type": first.get("type"),
            "question": first.get("question"),
            "ground_truth": first.get("ground_truth"),
            "scores": scores,
            "failure_modes_observed": dict(failure_modes),
            "n_runs": len(question_runs),
            "stability": classify_stability(rel_std),
        })
    return merged


def aggregate_metrics(merged: list[dict]) -> dict[str, dict[str, float]]:
    """Compute aggregate (across questions) median/mean/std for each metric.

    Each *question's* median goes into the aggregate computation — this is
    the headline number the gate compares against thresholds.
    """
    out: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        medians = [q["scores"][metric]["median"] for q in merged]
        out[metric] = {
            "median": round(_median(medians), 4),
            "mean": round(_mean(medians), 4),
            "std": round(_std(medians), 4),
        }
    return out


def stability_breakdown(merged: list[dict]) -> dict[str, int]:
    """Count of questions in each stability bucket."""
    counts = Counter(q["stability"] for q in merged)
    # Ensure all three keys exist for reporting consistency
    for k in ["stable", "mild-variance", "bimodal"]:
        counts.setdefault(k, 0)
    return dict(counts)


def majority_failure_modes(merged: list[dict]) -> Counter:
    """Aggregate failure-mode counts using each question's *majority* mode.

    Question Q06 with failure_modes {retrieval-miss: 1, synthesis-fail: 2}
    contributes 1 to synthesis-fail. Ties broken by sort order (deterministic).

    This gives us a single representative failure mode per question for the
    gate's count-based thresholds (retrieval_miss_max, none_min).
    """
    counts: Counter = Counter()
    for q in merged:
        modes = q["failure_modes_observed"]
        if not modes:
            continue
        # max-by-count, deterministic tiebreak via key
        winning = max(sorted(modes.items()), key=lambda kv: kv[1])
        counts[winning[0]] += 1
    return counts


def build_final_report(
    per_run_results: list[list[dict]],
    run_metadata: dict[str, Any],
    individual_run_ids: list[str],
) -> dict[str, Any]:
    """Assemble the full merged scored-run document.

    `run_metadata` is the config dict (top_k, retriever, models, etc.)
    shared by every run — passed in so this module stays I/O-free.
    """
    merged = merge_per_question(per_run_results)
    return {
        "format": "scored_n",         # so the gate can detect the new format
        "n_runs": len(per_run_results),
        "individual_run_ids": individual_run_ids,
        "config": run_metadata,
        "results": merged,
        "aggregates": aggregate_metrics(merged),
        "stability_breakdown": stability_breakdown(merged),
        "majority_failure_modes": dict(majority_failure_modes(merged)),
    }
