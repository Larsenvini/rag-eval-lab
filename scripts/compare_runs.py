"""Compare any scored run against baseline.json and show what moved.

Usage:
    python -m scripts.compare_runs evals/results/scored_<timestamp>.json
    python -m scripts.compare_runs evals/results/scored_<timestamp>.json --baseline evals/results/other.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

BASELINE_PATH = Path("evals/results/baseline.json")

METRICS = ["faithfulness", "answer_relevance", "ground_truth_similarity"]
METRIC_LABELS = {"faithfulness": "Faith.", "answer_relevance": "Rel.", "ground_truth_similarity": "GT Sim."}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def index_by_id(results: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in results}


def delta_color(d: float) -> str:
    if d >= 0.05:
        return "green"
    if d <= -0.05:
        return "red"
    return "dim"


def fmt_delta(d: float) -> str:
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:+.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a scored run against the baseline")
    parser.add_argument("experiment", help="Path to scored_<timestamp>.json to compare")
    parser.add_argument("--baseline", default=str(BASELINE_PATH), help="Baseline file to compare against")
    args = parser.parse_args()

    console = Console()

    exp_path = Path(args.experiment)
    base_path = Path(args.baseline)

    for p in (exp_path, base_path):
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            return 1

    exp_data = load(exp_path)
    base_data = load(base_path)

    exp_results = index_by_id(exp_data["results"])
    base_results = index_by_id(base_data["results"])

    exp_note = exp_data.get("note", "") or exp_data.get("run_id", "experiment")
    base_note = base_data.get("note", "") or base_data.get("run_id", "baseline")

    common_ids = sorted(set(exp_results) & set(base_results))
    if not common_ids:
        console.print("[red]No overlapping question IDs between the two files.[/red]")
        return 1

    # ── Aggregate comparison ────────────────────────────────────────────────
    console.print(f"\n[bold]Aggregate comparison[/bold]  baseline='{base_note}'  experiment='{exp_note}'  n={len(common_ids)}\n")

    agg_table = Table(show_header=True)
    agg_table.add_column("Metric")
    agg_table.add_column("Baseline", justify="right")
    agg_table.add_column("Experiment", justify="right")
    agg_table.add_column("Delta", justify="right")

    for m in METRICS:
        base_avg = sum(base_results[i].get(m, 0) for i in common_ids) / len(common_ids)
        exp_avg = sum(exp_results[i].get(m, 0) for i in common_ids) / len(common_ids)
        d = exp_avg - base_avg
        color = delta_color(d)
        agg_table.add_row(
            METRIC_LABELS[m],
            f"{base_avg:.3f}",
            f"{exp_avg:.3f}",
            f"[{color}]{fmt_delta(d)}[/{color}]",
        )

    console.print(agg_table)

    # ── Failure mode shift ──────────────────────────────────────────────────
    base_fm: dict[str, int] = {}
    exp_fm: dict[str, int] = {}
    for qid in common_ids:
        bf = base_results[qid].get("failure_mode", "unknown")
        ef = exp_results[qid].get("failure_mode", "unknown")
        base_fm[bf] = base_fm.get(bf, 0) + 1
        exp_fm[ef] = exp_fm.get(ef, 0) + 1

    all_modes = sorted(set(base_fm) | set(exp_fm))
    console.print("\n[bold]Failure mode shift:[/bold]")
    fm_table = Table(show_header=True)
    fm_table.add_column("Mode")
    fm_table.add_column("Baseline", justify="right")
    fm_table.add_column("Experiment", justify="right")
    fm_table.add_column("Delta", justify="right")

    for fm in all_modes:
        b = base_fm.get(fm, 0)
        e = exp_fm.get(fm, 0)
        d = e - b
        color = ("green" if fm == "none" else "red") if d != 0 else "dim"
        fm_table.add_row(
            fm,
            str(b),
            str(e),
            f"[{color}]{'+' if d > 0 else ''}{d}[/{color}]",
        )

    console.print(fm_table)

    # ── Per-question movers ─────────────────────────────────────────────────
    movers = []
    for qid in common_ids:
        b = base_results[qid]
        e = exp_results[qid]
        rel_delta = e.get("answer_relevance", 0) - b.get("answer_relevance", 0)
        faith_delta = e.get("faithfulness", 0) - b.get("faithfulness", 0)
        total_delta = rel_delta + faith_delta
        if abs(total_delta) >= 0.1:
            movers.append((qid, b, e, faith_delta, rel_delta, total_delta))

    movers.sort(key=lambda x: -abs(x[5]))

    if movers:
        console.print(f"\n[bold]Questions that moved >= 0.1 on faith+relevance ({len(movers)} total):[/bold]")
        mover_table = Table(show_header=True)
        mover_table.add_column("ID", style="dim")
        mover_table.add_column("Topic")
        mover_table.add_column("Type")
        mover_table.add_column("dFaith.", justify="right")
        mover_table.add_column("dRel.", justify="right")
        mover_table.add_column("Mode: base->exp")

        for qid, b, e, fd, rd, _ in movers:
            bfm = b.get("failure_mode", "?")
            efm = e.get("failure_mode", "?")
            mode_str = f"{bfm} -> {efm}" if bfm != efm else f"[dim]{bfm}[/dim]"
            mover_table.add_row(
                qid,
                b.get("topic", "?"),
                b.get("type", "?"),
                f"[{delta_color(fd)}]{fmt_delta(fd)}[/{delta_color(fd)}]",
                f"[{delta_color(rd)}]{fmt_delta(rd)}[/{delta_color(rd)}]",
                mode_str,
            )

        console.print(mover_table)
    else:
        console.print("\n[dim]No questions moved more than 0.1 on faith+relevance.[/dim]")

    # ── Canary check (Q06, Q20, Q25) ───────────────────────────────────────
    canaries = ["Q06", "Q20", "Q25"]
    canary_present = [c for c in canaries if c in exp_results and c in base_results]
    if canary_present:
        console.print("\n[bold]Canary questions (were 0.00 in baseline):[/bold]")
        for qid in canary_present:
            b = base_results[qid]
            e = exp_results[qid]
            b_rel = b.get("answer_relevance", 0)
            e_rel = e.get("answer_relevance", 0)
            status = "[green]FIXED[/green]" if e_rel > 0.3 else "[red]still failing[/red]"
            console.print(f"  {qid}  base_rel={b_rel:.2f}  exp_rel={e_rel:.2f}  {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
