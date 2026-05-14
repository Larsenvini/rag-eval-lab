"""Quality gate as a pytest.

Reads a scored eval run, checks it against `evals/thresholds.json`, fails
the test (and therefore CI) on any violation.

Supports both formats:
  - Legacy single-run: `scored_*.json` with results[i].faithfulness etc.
  - N-run merged:      `scored_n_*.json` with results[i].scores.faithfulness.median
                       — preferred. Gate uses per-question medians.

Usage:
    # Default: latest scored_n_*.json (fallback: scored_*.json) in evals/results/
    pytest tests/test_eval_gate.py -v

    # Or point at a specific file
    EVAL_GATE_FILE=evals/results/scored_n_20260513_103000.json pytest tests/test_eval_gate.py -v

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

METRICS = ["faithfulness", "answer_relevance", "ground_truth_similarity"]


# ─── Format detection & metric extraction ─────────────────────────────────

def _is_n_format(data: dict) -> bool:
    """N-run merged file: each result has a 'scores' dict with medians."""
    return data.get("format") == "scored_n" or (
        data.get("results") and isinstance(data["results"][0].get("scores"), dict)
    )


def _metric_value(result: dict, metric: str, is_n: bool) -> float:
    """Pull the gate-relevant value for `metric` from a result dict.

    For N-run format, that's the median. For legacy single-run, the raw score.
    """
    if is_n:
        return float(result.get("scores", {}).get(metric, {}).get("median", 0.0))
    return float(result.get(metric, 0.0))


def _failure_mode(result: dict, is_n: bool) -> str:
    """Single representative failure mode per question.

    For N-run, we use the majority-vote already computed at merge time
    (stored in result['failure_modes_observed'] as a count dict).
    For legacy, the single `failure_mode` field.
    """
    if is_n:
        modes = result.get("failure_modes_observed", {})
        if not modes:
            return "unknown"
        # Same tiebreak rule as merge_scored_runs.majority_failure_modes
        return max(sorted(modes.items()), key=lambda kv: kv[1])[0]
    return result.get("failure_mode", "unknown")


# ─── Fixtures ─────────────────────────────────────────────────────────────


def _resolve_scored_file() -> Path:
    """Pick which scored run to check.

    Priority:
      1. EVAL_GATE_FILE env var (CI sets this to the run it just produced)
      2. Newest scored_n_*.json (preferred — N-run merged format)
      3. Newest scored_*.json (legacy single-run format)
    """
    env = os.environ.get("EVAL_GATE_FILE")
    if env:
        return Path(env)

    # Prefer the N-run merged format if any exist
    n_candidates = sorted(
        RESULTS_DIR.glob("scored_n_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if n_candidates:
        return n_candidates[0]

    # Fall back to single-run files; exclude the n-merged ones already checked
    legacy_candidates = [
        p for p in sorted(
            RESULTS_DIR.glob("scored_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not p.name.startswith("scored_n_")
    ]
    if legacy_candidates:
        return legacy_candidates[0]

    pytest.skip(
        "No scored_*.json or scored_n_*.json in evals/results/. Run "
        "`python -m scripts.run_eval_n` (preferred) or `run_eval` + `score_eval` first."
    )


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
        display = path
    print(f"\n[eval-gate] Checking: {display}")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["__is_n_format"] = _is_n_format(data)
    if data["__is_n_format"]:
        print(f"[eval-gate] N-run format detected (n={data.get('n_runs', '?')})")
    else:
        print(f"[eval-gate] Single-run format detected (no variance data)")
    return data


@pytest.fixture(scope="module")
def aggregates(scored) -> dict:
    """Compute aggregate metrics.

    For N-format: prefer the precomputed `aggregates` block (median-of-medians).
    For legacy: compute mean across questions on the fly.
    """
    results = scored["results"]
    n = len(results)
    if n == 0:
        pytest.fail("Scored run has zero results.")

    is_n = scored["__is_n_format"]
    if is_n and "aggregates" in scored:
        return {
            m: scored["aggregates"][m]["median"] for m in METRICS
        } | {"n": n}

    # Legacy or missing aggregates block: compute on the fly
    return {
        m: sum(_metric_value(r, m, is_n) for r in results) / n
        for m in METRICS
    } | {"n": n}


# ─── Aggregate threshold checks ───────────────────────────────────────────


@pytest.mark.parametrize("metric", METRICS)
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
    is_n = scored["__is_n_format"]
    # For N-format with majority_failure_modes precomputed, use it.
    if is_n and "majority_failure_modes" in scored:
        return Counter(scored["majority_failure_modes"])
    return Counter(_failure_mode(r, is_n) for r in scored["results"])


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


# ─── Stability checks (N-run format only) ────────────────────────────────


def test_bimodal_question_count(scored, thresholds):
    """Number of bimodal questions must stay at or below the configured ceiling.

    Skipped for legacy single-run files (no stability data available) and for
    N=1 runs (no variance can be measured with one sample).
    """
    if not scored["__is_n_format"]:
        pytest.skip("Stability check requires N-run format (scored_n_*.json)")

    n_runs = scored.get("n_runs", 1)
    if n_runs < 2:
        pytest.skip(f"Stability check requires N >= 2 (got N={n_runs})")

    stability_cfg = thresholds.get("stability", {})
    ceiling = stability_cfg.get("bimodal_max", {}).get("value")
    if ceiling is None:
        pytest.skip("No stability.bimodal_max threshold configured")

    counts = scored.get("stability_breakdown", {})
    observed = counts.get("bimodal", 0)
    assert observed <= ceiling, (
        f"\n  bimodal question count: {observed}  exceeds ceiling {ceiling}.\n"
        f"  Note: {stability_cfg['bimodal_max'].get('note', '')}\n"
    )


# ─── Canary checks ────────────────────────────────────────────────────────


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

    For N-run format, the gate uses the *median* relevance across runs,
    which is robust to single-run stochasticity.
    """
    qid = spec["id"]
    floor = spec["min_relevance"]

    match = next((r for r in scored["results"] if r.get("id") == qid), None)
    assert match is not None, f"Canary {qid} not found in scored results."

    is_n = scored["__is_n_format"]
    observed = _metric_value(match, "answer_relevance", is_n)
    label = "median relevance" if is_n else "relevance"

    assert observed >= floor, (
        f"\n  Canary {qid} {label}: {observed:.2f}  is below floor {floor:.2f}.\n"
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
    prev_is_n = _is_n_format(prev)
    prev_results = prev.get("results", [])
    if not prev_results:
        pytest.skip("last_main.json has no results.")

    # If last_main is N-format and has its own aggregates block, use that.
    if prev_is_n and "aggregates" in prev:
        prev_agg = {m: prev["aggregates"][m]["median"] for m in METRICS}
    else:
        n = len(prev_results)
        prev_agg = {
            m: sum(_metric_value(r, m, prev_is_n) for r in prev_results) / n
            for m in METRICS
        }

    max_drop = thresholds["regression"]["max_drop_per_metric"]

    regressions = []
    for metric in METRICS:
        drop = prev_agg[metric] - aggregates[metric]
        if drop > max_drop:
            regressions.append(
                f"  {metric}: {prev_agg[metric]:.3f} -> {aggregates[metric]:.3f}  (drop {drop:.3f} > {max_drop:.2f})"
            )

    assert not regressions, "\nRegression vs last main:\n" + "\n".join(regressions)


def _failure_counts_from(data: dict) -> Counter:
    """Failure counts for any data dict (current or last_main), format-aware."""
    is_n = _is_n_format(data)
    if is_n and "majority_failure_modes" in data:
        return Counter(data["majority_failure_modes"])
    return Counter(_failure_mode(r, is_n) for r in data.get("results", []))


def test_no_failure_mode_drift_vs_last_main(scored, thresholds):
    """Failure-mode counts must not drift more than the configured amounts
    between consecutive main-branch runs.

    Absolute floors catch catastrophic regressions; this catches *changes*
    that absolute floors miss. Example: if `none` slowly drops from 8 to 4
    across several PRs, each individual drop may stay above the absolute
    floor while the trend is bad. This test catches that.
    """
    if not LAST_MAIN.exists():
        pytest.skip("No evals/results/last_main.json yet. First run; nothing to compare.")

    drift_cfg = thresholds.get("regression", {}).get("failure_mode_drift", {})
    if not drift_cfg:
        pytest.skip("No failure_mode_drift thresholds configured")

    prev = json.loads(LAST_MAIN.read_text(encoding="utf-8"))
    prev_counts = _failure_counts_from(prev)
    cur_counts = _failure_counts_from(scored)

    violations = []
    for mode, rule in drift_cfg.items():
        if mode.startswith("_"):
            continue
        prev_n = prev_counts.get(mode, 0)
        cur_n = cur_counts.get(mode, 0)

        max_increase = rule.get("max_increase")
        max_decrease = rule.get("max_decrease")

        if max_increase is not None:
            growth = cur_n - prev_n
            if growth > max_increase:
                violations.append(
                    f"  {mode}: {prev_n} -> {cur_n}  (+{growth}, max increase {max_increase})"
                )
        if max_decrease is not None:
            shrink = prev_n - cur_n
            if shrink > max_decrease:
                violations.append(
                    f"  {mode}: {prev_n} -> {cur_n}  (-{shrink}, max decrease {max_decrease})"
                )

    assert not violations, "\nFailure-mode drift vs last main:\n" + "\n".join(violations)


# ─── Code-check gates ─────────────────────────────────────────────────────


def _resolve_code_checks_file() -> Path | None:
    """Pick the newest code_checks_*.json. Returns None if none exist."""
    env = os.environ.get("CODE_CHECKS_FILE")
    if env:
        p = Path(env)
        return p if p.exists() else None

    candidates = sorted(
        RESULTS_DIR.glob("code_checks_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


@pytest.fixture(scope="module")
def code_checks() -> dict | None:
    path = _resolve_code_checks_file()
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "field,threshold_key",
    [
        ("missing_citation",      "missing_citation_max"),
        ("suspiciously_short",    "suspiciously_short_max"),
        ("has_forbidden_patterns","forbidden_patterns_max"),
    ],
    ids=["missing_citation", "suspiciously_short", "forbidden_patterns"],
)
def test_code_check_threshold(field, threshold_key, code_checks, thresholds):
    """Deterministic checks on RAG answers must stay within thresholds.

    Skipped if no code_checks_*.json file exists yet — generate one with:
        python -m scripts.code_checks evals/results/run_<ts>.json
    """
    if code_checks is None:
        pytest.skip("No code_checks_*.json file found. Run `scripts.code_checks` to generate one.")

    cfg = thresholds.get("code_checks", {})
    if not cfg or threshold_key not in cfg:
        pytest.skip(f"No code_checks.{threshold_key} threshold configured")

    ceiling = cfg[threshold_key]["value"]
    observed = code_checks.get(field, 0)
    assert observed <= ceiling, (
        f"\n  {field}: {observed}  exceeds ceiling {ceiling}.\n"
        f"  Note: {cfg[threshold_key].get('note', '')}\n"
    )
