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


def aggregate(results: list[dict]) -> dict[str, float]:
    n = len(results) or 1
    return {key: sum(r.get(key, 0) for r in results) / n for key, _ in METRICS}


def failure_breakdown(results: list[dict]) -> Counter:
    return Counter(r.get("failure_mode", "unknown") for r in results)


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


def render(scored: dict, baseline: dict | None, last_main: dict | None) -> str:
    results = scored.get("results", [])
    if not results:
        return "⚠️ Eval run had no results."

    n = len(results)
    cur = aggregate(results)
    cur_fm = failure_breakdown(results)

    # Header & meta
    run_id = scored.get("run_id", "?")
    note = scored.get("note") or ""
    cfg = scored.get("config", {})
    retriever = cfg.get("retriever", "?")
    top_k = cfg.get("top_k", "?")
    gen_model = cfg.get("generation_model", "?")

    lines: list[str] = []
    lines.append("## 🤖 RAG eval results")
    lines.append("")
    lines.append(f"**Run:** `{run_id}` · **n** = {n} · **retriever:** `{retriever}` · "
                 f"**top_k:** `{top_k}` · **model:** `{gen_model}`")
    if note:
        lines.append(f"_{note}_")
    lines.append("")

    # ── Aggregate table ──────────────────────────────────────────────────
    base = aggregate(baseline["results"]) if baseline else None
    last = aggregate(last_main["results"]) if last_main else None

    lines.append("### Aggregate scores")
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
    base_fm = failure_breakdown(baseline["results"]) if baseline else Counter()
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

    # ── Canaries (if labeled in thresholds, we don't know here — surface the three best-known) ──
    canary_ids = {"Q06", "Q20", "Q25"}
    canary_rows = []
    for r in results:
        if r.get("id") in canary_ids:
            canary_rows.append((r["id"], r.get("answer_relevance", 0), r.get("failure_mode", "?")))

    if canary_rows:
        lines.append("### Canaries")
        lines.append("")
        lines.append("| ID | Relevance | Mode |")
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
            d_rel   = r.get("answer_relevance", 0) - base_by_id[qid].get("answer_relevance", 0)
            d_faith = r.get("faithfulness", 0)    - base_by_id[qid].get("faithfulness", 0)
            movers.append((qid, d_faith, d_rel,
                          base_by_id[qid].get("failure_mode", "?"),
                          r.get("failure_mode", "?")))
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

    lines.append("---")
    lines.append("_Generated by [`scripts/post_eval_summary.py`](scripts/post_eval_summary.py)_")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Format a scored eval as Markdown")
    parser.add_argument("scored_file", help="Path to scored_*.json")
    parser.add_argument("--baseline", default="evals/results/baseline.json")
    parser.add_argument("--last-main", default="evals/results/last_main.json")
    args = parser.parse_args()

    scored = load_json(Path(args.scored_file))
    if not scored:
        print(f"Scored file not found: {args.scored_file}", file=sys.stderr)
        return 1

    baseline = load_json(Path(args.baseline))
    last_main = load_json(Path(args.last_main))

    print(render(scored, baseline, last_main))
    return 0


if __name__ == "__main__":
    sys.exit(main())
