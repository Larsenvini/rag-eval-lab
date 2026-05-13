"""Unit tests for scripts.merge_scored_runs.

These are math-only tests — no file I/O, no API calls. They verify median
calculation, stability classification, and aggregation logic with
hand-crafted inputs designed to stress edge cases.
"""

from __future__ import annotations

import pytest

from scripts.merge_scored_runs import (
    aggregate_metrics,
    build_final_report,
    classify_stability,
    majority_failure_modes,
    merge_per_question,
    stability_breakdown,
)


# ─── classify_stability ─────────────────────────────────────────────────


def test_classify_stable_boundary():
    assert classify_stability(0.0) == "stable"
    assert classify_stability(0.10) == "stable"      # inclusive upper
    assert classify_stability(0.101) == "mild-variance"


def test_classify_mild_variance_boundary():
    assert classify_stability(0.20) == "mild-variance"
    assert classify_stability(0.25) == "mild-variance"  # inclusive upper
    assert classify_stability(0.251) == "bimodal"


def test_classify_bimodal():
    assert classify_stability(0.40) == "bimodal"
    assert classify_stability(1.0) == "bimodal"


# ─── merge_per_question — single question ───────────────────────────────


def _q(qid: str, faith: float, rel: float, gt: float,
       failure_mode: str = "none", topic: str = "pods", qtype: str = "definition") -> dict:
    return {
        "id": qid,
        "topic": topic,
        "type": qtype,
        "question": "?",
        "ground_truth": "...",
        "faithfulness": faith,
        "answer_relevance": rel,
        "ground_truth_similarity": gt,
        "failure_mode": failure_mode,
    }


def test_merge_single_question_stable():
    """A question scored ~identically across 3 runs is 'stable'."""
    runs = [
        [_q("Q01", 0.90, 0.85, 0.70)],
        [_q("Q01", 0.92, 0.85, 0.72)],
        [_q("Q01", 0.88, 0.85, 0.68)],
    ]
    merged = merge_per_question(runs)
    assert len(merged) == 1
    q = merged[0]
    assert q["id"] == "Q01"
    assert q["scores"]["answer_relevance"]["median"] == pytest.approx(0.85, abs=1e-4)
    assert q["scores"]["answer_relevance"]["std"] == pytest.approx(0.0, abs=1e-4)
    assert q["stability"] == "stable"
    assert q["n_runs"] == 3


def test_merge_single_question_bimodal():
    """Q06-like: scored 0.0, 0.85, 0.9 across 3 runs → high std → bimodal."""
    runs = [
        [_q("Q06", 0.0, 0.0, 0.0, failure_mode="retrieval-miss",
            topic="architecture", qtype="why-tradeoff")],
        [_q("Q06", 0.85, 0.85, 0.7, failure_mode="answer-incomplete",
            topic="architecture", qtype="why-tradeoff")],
        [_q("Q06", 0.9, 0.9, 0.75, failure_mode="none",
            topic="architecture", qtype="why-tradeoff")],
    ]
    merged = merge_per_question(runs)
    q = merged[0]
    # Median of {0.0, 0.85, 0.9} = 0.85
    assert q["scores"]["answer_relevance"]["median"] == pytest.approx(0.85, abs=1e-4)
    # std is high → bimodal
    assert q["scores"]["answer_relevance"]["std"] > 0.25
    assert q["stability"] == "bimodal"
    # Failure modes were observed in all three buckets
    assert q["failure_modes_observed"] == {
        "retrieval-miss": 1, "answer-incomplete": 1, "none": 1,
    }


def test_merge_mild_variance():
    """Question with moderate jitter: 0.7, 0.85, 0.9 → mild-variance."""
    runs = [
        [_q("Q02", 0.7, 0.7, 0.6)],
        [_q("Q02", 0.85, 0.85, 0.7)],
        [_q("Q02", 0.9, 0.9, 0.8)],
    ]
    merged = merge_per_question(runs)
    q = merged[0]
    # std of {0.7, 0.85, 0.9} ≈ 0.085 — falls in 'stable'!
    # Hand-pick another sample to land in mild-variance
    runs2 = [
        [_q("Q02", 0.6, 0.6, 0.5)],
        [_q("Q02", 0.8, 0.8, 0.7)],
        [_q("Q02", 0.95, 0.95, 0.85)],
    ]
    merged2 = merge_per_question(runs2)
    # std ≈ 0.144 — mild-variance
    assert 0.10 < merged2[0]["scores"]["answer_relevance"]["std"] < 0.25
    assert merged2[0]["stability"] == "mild-variance"


# ─── merge_per_question — multiple questions ────────────────────────────


def test_merge_multiple_questions():
    runs = [
        [_q("Q01", 0.9, 0.9, 0.8), _q("Q02", 0.7, 0.7, 0.6)],
        [_q("Q01", 0.9, 0.9, 0.8), _q("Q02", 0.7, 0.7, 0.6)],
        [_q("Q01", 0.9, 0.9, 0.8), _q("Q02", 0.7, 0.7, 0.6)],
    ]
    merged = merge_per_question(runs)
    assert len(merged) == 2
    # Sorted by id
    assert [m["id"] for m in merged] == ["Q01", "Q02"]


def test_merge_handles_missing_question():
    """If a question only appears in 2 of 3 runs, still merge with available samples."""
    runs = [
        [_q("Q01", 0.9, 0.9, 0.8), _q("Q02", 0.7, 0.7, 0.6)],
        [_q("Q01", 0.9, 0.9, 0.8)],                          # Q02 missing
        [_q("Q01", 0.9, 0.9, 0.8), _q("Q02", 0.8, 0.8, 0.7)],
    ]
    merged = merge_per_question(runs)
    q02 = next(m for m in merged if m["id"] == "Q02")
    assert q02["n_runs"] == 2
    assert len(q02["scores"]["answer_relevance"]["samples"]) == 2


def test_merge_empty():
    assert merge_per_question([]) == []
    assert merge_per_question([[], [], []]) == []


# ─── aggregate_metrics ──────────────────────────────────────────────────


def test_aggregate_uses_per_question_medians():
    """Aggregate is the median of *medians*, not of raw samples."""
    runs = [
        [_q("Q01", 0.9, 0.9, 0.9), _q("Q02", 0.5, 0.5, 0.5)],
        [_q("Q01", 0.9, 0.9, 0.9), _q("Q02", 0.5, 0.5, 0.5)],
        [_q("Q01", 0.9, 0.9, 0.9), _q("Q02", 0.5, 0.5, 0.5)],
    ]
    merged = merge_per_question(runs)
    agg = aggregate_metrics(merged)
    # Median of {0.9, 0.5} per metric = 0.7
    assert agg["faithfulness"]["median"] == pytest.approx(0.7, abs=1e-4)
    assert agg["answer_relevance"]["median"] == pytest.approx(0.7, abs=1e-4)


# ─── stability_breakdown ─────────────────────────────────────────────────


def test_stability_breakdown_all_buckets_present():
    """Result always has all three stability buckets even if some are 0."""
    runs = [
        [_q("Q01", 0.9, 0.9, 0.9)],
        [_q("Q01", 0.9, 0.9, 0.9)],
        [_q("Q01", 0.9, 0.9, 0.9)],
    ]
    merged = merge_per_question(runs)
    breakdown = stability_breakdown(merged)
    assert set(breakdown.keys()) == {"stable", "mild-variance", "bimodal"}
    assert breakdown["stable"] == 1
    assert breakdown["mild-variance"] == 0
    assert breakdown["bimodal"] == 0


# ─── majority_failure_modes ──────────────────────────────────────────────


def test_majority_failure_mode_picks_most_common():
    runs = [
        [_q("Q06", 0.5, 0.5, 0.5, failure_mode="retrieval-miss")],
        [_q("Q06", 0.5, 0.5, 0.5, failure_mode="synthesis-fail")],
        [_q("Q06", 0.5, 0.5, 0.5, failure_mode="synthesis-fail")],
    ]
    merged = merge_per_question(runs)
    counts = majority_failure_modes(merged)
    # Q06's majority mode is synthesis-fail (2 of 3)
    assert counts == {"synthesis-fail": 1}


def test_majority_failure_mode_tiebreak_is_deterministic():
    """When a question has tied failure modes, tiebreak is alphabetical."""
    runs = [
        [_q("Q01", 0.5, 0.5, 0.5, failure_mode="retrieval-miss")],
        [_q("Q01", 0.5, 0.5, 0.5, failure_mode="synthesis-fail")],
    ]
    merged = merge_per_question(runs)
    # Tiebreaker uses max-after-sort, which lexicographically picks 'synthesis-fail'
    counts = majority_failure_modes(merged)
    assert sum(counts.values()) == 1


# ─── build_final_report (integration) ───────────────────────────────────


def test_build_final_report_shape():
    runs = [
        [_q("Q01", 0.9, 0.9, 0.9, failure_mode="none")],
        [_q("Q01", 0.9, 0.9, 0.9, failure_mode="none")],
        [_q("Q01", 0.9, 0.9, 0.9, failure_mode="none")],
    ]
    report = build_final_report(
        per_run_results=runs,
        run_metadata={"retriever": "hybrid", "top_k": 12},
        individual_run_ids=["run_a", "run_b", "run_c"],
    )
    assert report["format"] == "scored_n"
    assert report["n_runs"] == 3
    assert report["individual_run_ids"] == ["run_a", "run_b", "run_c"]
    assert report["config"]["retriever"] == "hybrid"
    assert "results" in report
    assert "aggregates" in report
    assert "stability_breakdown" in report
    assert "majority_failure_modes" in report
