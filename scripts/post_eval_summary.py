"""Format a scored eval run as Markdown.

The CI workflow runs this and posts the output as a PR comment so reviewers
can see the eval impact of their change without leaving the GitHub UI.

Usage:
    python -m scripts.post_eval_summary <scored_file> [--baseline path] [--last-main path]
    # Writes Markdown to stdout. CI redirects it into a file then posts via gh CLI.

Outputs a self-contained Markdown block:
    - aggregate scores table (current vs baseline vs last main)
    - failure mode counts
    - canary status
    - top 5 movers vs baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


METRICS = [
    ("faithfulness",            "Faithfulness"),
    ("answer_relevance",        "Relevance"),
    ("ground_truth_similarity", "GT Sim."),
]


def load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _is_n_format(data: dict | None) -> bool:
    if not data:
        return False
    return data.get("format") == "scored_n" or (
        data.get("results") and isinstance(data["results"][0].get("scores"), dict)
    )


def _metric_value(result: dict, metric: str, is_n: bool) -> float:
    if is_n:
        return float(result.get("scores", {}).get(metric, {}).get("median", 0.0))
    return float(result.get(metric, 0.0))


def _failure_mode_for(result: dict, is_n: bool) -> str:
    if is_n:
        modes = result.get("failure_modes_observed", {})
        if not modes:
            return "unknown"
        return max(sorted(modes.items()), key=lambda kv: kv[1])[0]
    return result.get("failure_mode", "unknown")


def aggregate(results: list[dict], is_n: bool = False) -> dict[str, float]:
    n = len(results) or 1
    return {
        key: sum(_metric_value(r, key, is_n) for r in results) / n
        for key, _ in METRICS
    }


def failure_breakdown(results: list[dict], is_n: bool = False) -> Counter:
    return Counter(_failure_mode_for(r, is_n) for r in results)


def fmt_delta(delta: float) -> str:
    if delta >= 0.05:  return f"🟢 +{delta:.2f}"
    if delta <= -0.05: return f"🔴 {delta:.2f}"
    return f"⚪ {delta:+.2f}"


def fmt_count_delta(delta: int, lower_is_better: bool = False) -> str:
    if delta == 0: return "⚪ 0"
    good = (delta < 0) if lower_is_better else (delta > 0)
    emoji = "🟢" if good else "🔴"
    sign = "+" if delta > 0 else ""
    return f"{emoji} {sign}{delta}"


def render(scored: dict, baseline: dict | None, last_main: dict | None,
           code_checks: dict | None = None) -> str:
    results = scored.get("results", [])
    if not results:
        return "⚠️ Eval run had no results."

    is_n = _is_n_format(scored)
    base_is_n = _is_n_format(baseline)
    last_is_n = _is_n_format(last_main)

    n = len(results)
    cur = aggregate(results, is_n=is_n)
    cur_fm = failure_breakdown(results, is_n=is_n)

    # Header & meta
    run_id = scored.get("run_id", "?")
    note = scored.get("note") or ""
    cfg = scored.get("config", {})
    retriever = cfg.get("retriever", "?")
    top_k = cfg.get("top_k", "?")
    gen_model = cfg.get("generation_model", "?")
    n_runs = scored.get("n_runs")

    lines: list[str] = []
    lines.append("## 🤖 RAG eval results")
    lines.append("")
    header_meta = f"**Run:** `{run_id}` · **questions** = {n} · **retriever:** `{retriever}` · " \
                  f"**top_k:** `{top_k}` · **model:** `{gen_model}`"
    if n_runs:
        header_meta += f" · **N runs** = `{n_runs}` (median-aggregated)"
    lines.append(header_meta)
    if note:
        lines.append(f"_{note}_")
    lines.append("")

    # ── Aggregate table ──────────────────────────────────────────────────
    base = aggregate(baseline["results"], is_n=base_is_n) if baseline else None
    last = aggregate(last_main["results"], is_n=last_is_n) if last_main else None

    lines.append("### Aggregate scores" + (" (medians across N runs)" if is_n else ""))
    lines.append("")
    header = "| Metric |   This run | vs baseline | vs last main |"
    sep    = "|--------|-----------:|------------:|-------------:|"
    lines.append(header)
    lines.append(sep)
    for key, label in METRICS:
        v = cur[key]
        base_delta = fmt_delta(v - base[key]) if base else "—"
        last_delta = fmt_delta(v - last[key]) if last else "—"
        lines.append(f"| **{label}** | {v:.3f} | {base_delta} | {last_delta} |")
    lines.append("")

    # ── Failure mode breakdown ───────────────────────────────────────────
    base_fm = failure_breakdown(baseline["results"], is_n=base_is_n) if baseline else Counter()
    all_modes = sorted(set(cur_fm) | set(base_fm),
                       key=lambda m: -cur_fm.get(m, 0))

    # Modes where "more is better" — `none` (clean wins).
    UP_IS_GOOD = {"none"}

    lines.append("### Failure modes")
    lines.append("")
    lines.append("| Mode | This run | vs baseline |")
    lines.append("|------|---------:|------------:|")
    for mode in all_modes:
        cur_c = cur_fm.get(mode, 0)
        base_c = base_fm.get(mode, 0)
        delta = cur_c - base_c
        lower_better = mode not in UP_IS_GOOD
        lines.append(f"| `{mode}` | {cur_c} | {fmt_count_delta(delta, lower_better)} |")
    lines.append("")

    # ── Stability (N-run only, and only meaningful when N >= 2) ─────────
    if is_n and (n_runs or 0) >= 2:
        breakdown = scored.get("stability_breakdown", {})
        if breakdown:
            stable = breakdown.get("stable", 0)
            mild = breakdown.get("mild-variance", 0)
            bimodal = breakdown.get("bimodal", 0)
            lines.append("### Stability across N runs")
            lines.append("")
            lines.append("| Bucket | Count |")
            lines.append("|--------|------:|")
            lines.append(f"| 🟢 stable (std ≤ 0.10)         | {stable} |")
            lines.append(f"| 🟡 mild-variance (≤ 0.25)      | {mild} |")
            lines.append(f"| 🔴 bimodal (> 0.25)            | {bimodal} |")
            lines.append("")
            # Surface the actual bimodal question IDs — useful for debugging
            bimodals = [q["id"] for q in results if q.get("stability") == "bimodal"]
            if bimodals:
                lines.append(f"_Bimodal questions:_ {', '.join(f'`{b}`' for b in sorted(bimodals))}")
                lines.append("")
    elif is_n and (n_runs or 0) < 2:
        lines.append("_Stability check skipped: only 1 run (need ≥2 for variance). "
                     "Main-branch pushes run N=3 for the canonical record._")
        lines.append("")

    # ── Canaries (if labeled in thresholds, we don't know here — surface the three best-known) ──
    canary_ids = {"Q06", "Q20", "Q25"}
    canary_rows = []
    for r in results:
        if r.get("id") in canary_ids:
            rel = _metric_value(r, "answer_relevance", is_n)
            mode = _failure_mode_for(r, is_n)
            canary_rows.append((r["id"], rel, mode))

    if canary_rows:
        lines.append("### Canaries")
        lines.append("")
        label = "Median Relevance" if is_n else "Relevance"
        lines.append(f"| ID | {label} | Mode |")
        lines.append("|----|----------:|------|")
        for qid, rel, mode in sorted(canary_rows):
            emoji = "🟢" if rel >= 0.5 else ("🟡" if rel >= 0.2 else "🔴")
            lines.append(f"| **{qid}** | {emoji} {rel:.2f} | `{mode}` |")
        lines.append("")

    # ── Top movers vs baseline (relevance delta) ─────────────────────────
    if baseline:
        base_by_id = {r["id"]: r for r in baseline["results"] if "id" in r}
        movers: list[tuple[str, float, float, str, str]] = []
        for r in results:
            qid = r.get("id")
            if not qid or qid not in base_by_id:
                continue
            d_rel = _metric_value(r, "answer_relevance", is_n) - _metric_value(base_by_id[qid], "answer_relevance", base_is_n)
            d_faith = _metric_value(r, "faithfulness", is_n) - _metric_value(base_by_id[qid], "faithfulness", base_is_n)
            movers.append((qid, d_faith, d_rel,
                          _failure_mode_for(base_by_id[qid], base_is_n),
                          _failure_mode_for(r, is_n)))
        # Sort by |d_rel| descending
        movers.sort(key=lambda t: -abs(t[2]))

        top = movers[:5]
        if top:
            lines.append("### Top 5 movers (by |Δ relevance|)")
            lines.append("")
            lines.append("| ID | ΔFaith | ΔRel | base → exp |")
            lines.append("|----|-------:|-----:|-----------|")
            for qid, df, dr, bm, em in top:
                lines.append(f"| **{qid}** | {df:+.2f} | {dr:+.2f} | `{bm}` → `{em}` |")
            lines.append("")

    # ── Code checks (deterministic, no LLM) ──────────────────────────────
    if code_checks:
        lines.append("### Code checks (deterministic)")
        lines.append("")
        lines.append("| Check | Count | Rate |")
        lines.append("|-------|------:|-----:|")
        n_total = code_checks.get("n", 0) or 1
        refusals = code_checks.get("refusals", 0)
        miss_cit = code_checks.get("missing_citation", 0)
        short = code_checks.get("suspiciously_short", 0)
        forbidden = code_checks.get("has_forbidden_patterns", 0)
        all_pass = code_checks.get("all_checks_pass", 0)
        lines.append(f"| Refusals           | {refusals} | {refusals/n_total:.0%} |")
        lines.append(f"| Missing citation   | {miss_cit} | {miss_cit/n_total:.0%} |")
        lines.append(f"| Suspiciously short | {short} | {short/n_total:.0%} |")
        lines.append(f"| Forbidden patterns | {forbidden} | {forbidden/n_total:.0%} |")
        lines.append(f"| **All checks pass** | **{all_pass}** | **{all_pass/n_total:.0%}** |")
        median_words = code_checks.get("median_word_count_non_refusal", 0)
        lines.append("")
        lines.append(f"_Median word count on non-refusal answers: **{median_words}**_")
        lines.append("")

    lines.append("---")
    lines.append("_Generated by [`scripts/post_eval_summary.py`](scripts/post_eval_summary.py)_")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Format a scored eval as Markdown")
    parser.add_argument("scored_file", help="Path to scored_*.json")
    parser.add_argument("--baseline", default="evals/results/baseline.json")
    parser.add_argument("--last-main", default="evals/results/last_main.json")
    parser.add_argument("--code-checks", default=None,
                        help="Path to code_checks_*.json (optional). If omitted, "
                             "tries to find the newest one in evals/results/.")
    args = parser.parse_args()

    scored = load_json(Path(args.scored_file))
    if not scored:
        print(f"Scored file not found: {args.scored_file}", file=sys.stderr)
        return 1

    baseline = load_json(Path(args.baseline))
    last_main = load_json(Path(args.last_main))

    # Code checks: explicit path > newest in results dir
    code_checks: dict | None = None
    if args.code_checks:
        code_checks = load_json(Path(args.code_checks))
    else:
        candidates = sorted(
            Path("evals/results").glob("code_checks_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            code_checks = load_json(candidates[0])

    print(render(scored, baseline, last_main, code_checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
