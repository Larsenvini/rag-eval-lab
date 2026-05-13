"""Quality gate as a pytest.

Reads a scored eval run, checks it against `evals/thresholds.json`, fails
the test (and therefore CI) on any violation.

Usage:
    # Default: latest scored_*.json in evals/results/
    pytest tests/test_eval_gate.py -v

    # Or point at a specific file
    EVAL_GATE_FILE=evals/results/scored_20260511_104839.json pytest tests/test_eval_gate.py -v

Why pytest and not a custom script:
    pytest gives us per-check pass/fail granularity in the CI output,
    nice red/green reporting, and a familiar interface to QA-minded engineers.
    Every threshold becomes its own test — easy to read which one failed.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pytest


# ─── Paths ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
THRESHOLDS = REPO / "evals" / "thresholds.json"
RESULTS_DIR = REPO / "evals" / "results"
LAST_MAIN = RESULTS_DIR / "last_main.json"


# ─── Fixtures ─────────────────────────────────────────────────────────────


def _resolve_scored_file() -> Path:
    """Pick which scored run to check.

    Priority:
      1. EVAL_GATE_FILE env var (CI sets this to the run it just produced)
      2. The newest scored_*.json in evals/results/ (local dev)
    """
    env = os.environ.get("EVAL_GATE_FILE")
    if env:
        return Path(env)

    candidates = sorted(
        RESULTS_DIR.glob("scored_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        pytest.skip(
            "No scored_*.json files in evals/results/. Run "
            "`python -m scripts.run_eval && python -m scripts.score_eval ...` first."
        )
    return candidates[0]


@pytest.fixture(scope="module")
def thresholds() -> dict:
    if not THRESHOLDS.exists():
        pytest.skip(f"No thresholds file at {THRESHOLDS}")
    return json.loads(THRESHOLDS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scored() -> dict:
    path = _resolve_scored_file().resolve()
    try:
        display = path.relative_to(REPO)
    except ValueError:
        display = path  # file is outside the repo; just show the full path
    print(f"\n[eval-gate] Checking: {display}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def aggregates(scored) -> dict:
    """Compute aggregate metrics from the scored results."""
    results = scored["results"]
    n = len(results)
    if n == 0:
        pytest.fail("Scored run has zero results.")
    return {
        "faithfulness": sum(r.get("faithfulness", 0) for r in results) / n,
        "answer_relevance": sum(r.get("answer_relevance", 0) for r in results) / n,
        "ground_truth_similarity": sum(r.get("ground_truth_similarity", 0) for r in results) / n,
        "n": n,
    }


# ─── Aggregate threshold checks ───────────────────────────────────────────


@pytest.mark.parametrize("metric", ["faithfulness", "answer_relevance", "ground_truth_similarity"])
def test_aggregate_above_threshold(metric, aggregates, thresholds):
    """Each aggregate metric must be at or above its floor."""
    floor = thresholds["aggregate"][metric]["min"]
    observed = aggregates[metric]
    assert observed >= floor, (
        f"\n  {metric}: {observed:.3f}  is below threshold {floor:.3f}.\n"
        f"  Note: {thresholds['aggregate'][metric].get('note', '')}\n"
    )


# ─── Failure-mode checks ──────────────────────────────────────────────────


def _failure_counts(scored) -> Counter:
    return Counter(r.get("failure_mode", "unknown") for r in scored["results"])


def test_retrieval_miss_count_below_ceiling(scored, thresholds):
    """retrieval-miss count must not exceed the configured ceiling."""
    counts = _failure_counts(scored)
    observed = counts.get("retrieval-miss", 0)
    ceiling = thresholds["failure_modes"]["retrieval_miss_max"]["value"]
    assert observed <= ceiling, (
        f"\n  retrieval-miss count: {observed}  exceeds ceiling {ceiling}.\n"
        f"  Note: {thresholds['failure_modes']['retrieval_miss_max'].get('note', '')}\n"
    )


def test_clean_answer_count_above_floor(scored, thresholds):
    """'none' (clean answer) count must be at or above the configured floor."""
    counts = _failure_counts(scored)
    observed = counts.get("none", 0)
    floor = thresholds["failure_modes"]["none_min"]["value"]
    assert observed >= floor, (
        f"\n  clean-answer count: {observed}  is below floor {floor}.\n"
        f"  Note: {thresholds['failure_modes']['none_min'].get('note', '')}\n"
    )


# ─── Canary checks ────────────────────────────────────────────────────────


def _canary_specs(thresholds) -> list[dict]:
    return thresholds["canaries"]["questions"]


@pytest.mark.parametrize(
    "spec",
    json.loads(THRESHOLDS.read_text(encoding="utf-8"))["canaries"]["questions"] if THRESHOLDS.exists() else [],
    ids=lambda s: s["id"],
)
def test_canary_question(spec, scored):
    """Each canary question's relevance must be at or above its minimum.

    Canaries are questions that score 0.00 in baseline and are *only* fixed
    by working retrieval. If a canary regresses, retrieval is broken — even
    if aggregates look fine.
    """
    qid = spec["id"]
    floor = spec["min_relevance"]

    match = next((r for r in scored["results"] if r.get("id") == qid), None)
    assert match is not None, f"Canary {qid} not found in scored results."

    observed = match.get("answer_relevance", 0)
    assert observed >= floor, (
        f"\n  Canary {qid} relevance: {observed:.2f}  is below floor {floor:.2f}.\n"
        f"  This question is a known retrieval-sensitive case. Aggregate scores "
        f"may still look fine, but core retrieval has regressed.\n"
    )


# ─── Regression check vs last main ────────────────────────────────────────


def test_no_regression_vs_last_main(aggregates, thresholds):
    """No aggregate may drop more than max_drop_per_metric from the previous
    main-branch scored run.

    Skipped if no last_main.json exists yet (first-ever run).
    """
    if not LAST_MAIN.exists():
        pytest.skip("No evals/results/last_main.json yet. First run; nothing to compare.")

    prev = json.loads(LAST_MAIN.read_text(encoding="utf-8"))
    prev_results = prev.get("results", [])
    if not prev_results:
        pytest.skip("last_main.json has no results.")

    n = len(prev_results)
    prev_agg = {
        "faithfulness": sum(r.get("faithfulness", 0) for r in prev_results) / n,
        "answer_relevance": sum(r.get("answer_relevance", 0) for r in prev_results) / n,
        "ground_truth_similarity": sum(r.get("ground_truth_similarity", 0) for r in prev_results) / n,
    }
    max_drop = thresholds["regression"]["max_drop_per_metric"]

    regressions = []
    for metric in ["faithfulness", "answer_relevance", "ground_truth_similarity"]:
        drop = prev_agg[metric] - aggregates[metric]
        if drop > max_drop:
            regressions.append(
                f"  {metric}: {prev_agg[metric]:.3f} -> {aggregates[metric]:.3f}  (drop {drop:.3f} > {max_drop:.2f})"
            )

    assert not regressions, "\nRegression vs last main:\n" + "\n".join(regressions)
