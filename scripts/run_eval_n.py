"""Run the full eval pipeline N times and merge the results.

The point: a single eval run is noisy because the LLM-as-judge is stochastic.
Running N times and taking the per-question median produces a stable score
plus a *measurable* variance estimate.

Usage:
    python -m scripts.run_eval_n                          # N=3 (default)
    python -m scripts.run_eval_n --n 5                    # more rigorous
    python -m scripts.run_eval_n --n 3 --note "post-fix"  # tag the run

Output:
    evals/results/scored_n_<timestamp>.json  — the merged report (what the gate reads)
    evals/results/run_<ts>.json, scored_<ts>.json
        — every individual eval+score is preserved for audit
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from scripts.merge_scored_runs import build_final_report


RESULTS_DIR = Path("evals/results")


def _newest_matching(pattern: str, since: float) -> Path | None:
    """Find the newest file matching pattern that was created after `since`."""
    candidates = [
        p for p in RESULTS_DIR.glob(pattern)
        if p.stat().st_mtime >= since
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_subprocess(cmd: list[str], console: Console) -> int:
    """Run a subprocess, streaming stdout/stderr live to the console."""
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    return subprocess.run(cmd, check=False).returncode


def run_one(note: str, console: Console) -> tuple[Path, Path] | None:
    """Run a single eval + score cycle. Returns (run_file, scored_file)."""
    python = sys.executable
    before = time.time()

    rc = _run_subprocess(
        [python, "-m", "scripts.run_eval", "--note", note],
        console,
    )
    if rc != 0:
        console.print("[red]run_eval failed[/red]")
        return None

    run_file = _newest_matching("run_*.json", before)
    if not run_file:
        console.print("[red]Could not find newly-created run_*.json file[/red]")
        return None

    rc = _run_subprocess(
        [python, "-m", "scripts.score_eval", str(run_file)],
        console,
    )
    if rc != 0:
        console.print(f"[red]score_eval failed for {run_file.name}[/red]")
        return None

    scored_file = _newest_matching("scored_*.json", before)
    if not scored_file:
        console.print("[red]Could not find newly-created scored_*.json file[/red]")
        return None

    return run_file, scored_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval N times and merge results")
    parser.add_argument("--n", type=int, default=3, help="Number of runs (default: 3)")
    parser.add_argument("--note", default="", help="Tag for the run, propagated to each iteration")
    args = parser.parse_args()

    if args.n < 1:
        print("--n must be >= 1", file=sys.stderr)
        return 1

    console = Console()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    console.print(Rule(f"[bold cyan]Running eval N={args.n} times[/bold cyan]"))

    individual_run_ids: list[str] = []
    per_run_results: list[list[dict]] = []
    shared_config: dict = {}

    for i in range(1, args.n + 1):
        console.print(Rule(f"[bold]Iteration {i}/{args.n}[/bold]"))
        iteration_note = f"{args.note} [n{i}/{args.n}]" if args.note else f"[n{i}/{args.n}]"
        result = run_one(iteration_note, console)
        if result is None:
            console.print(f"[red]Iteration {i} failed — aborting[/red]")
            return 1

        run_file, scored_file = result
        scored_data = json.loads(scored_file.read_text(encoding="utf-8"))

        individual_run_ids.append(scored_data.get("run_id", run_file.stem))
        per_run_results.append(scored_data.get("results", []))
        # Capture config from the first run; assume it's identical across runs
        if not shared_config:
            shared_config = scored_data.get("config", {})

    # Build the merged report
    console.print(Rule("[bold cyan]Merging results[/bold cyan]"))
    report = build_final_report(
        per_run_results=per_run_results,
        run_metadata=shared_config,
        individual_run_ids=individual_run_ids,
    )
    if args.note:
        report["note"] = args.note

    # Save it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"scored_n_{timestamp}.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Summary printout
    agg = report["aggregates"]
    stab = report["stability_breakdown"]
    fm = report["majority_failure_modes"]
    console.print()
    console.print(f"[green]✓ Merged {args.n} runs → {out_path}[/green]")
    console.print()
    console.print(f"[bold]Aggregates (medians of per-question medians):[/bold]")
    for metric, vals in agg.items():
        console.print(f"  {metric:<26} median={vals['median']:.3f}  "
                      f"mean={vals['mean']:.3f}  std={vals['std']:.3f}")
    console.print()
    console.print(f"[bold]Stability:[/bold] "
                  f"[green]{stab['stable']} stable[/green] · "
                  f"[yellow]{stab['mild-variance']} mild-variance[/yellow] · "
                  f"[red]{stab['bimodal']} bimodal[/red]")
    console.print(f"[bold]Failure modes (majority):[/bold] {dict(fm)}")
    console.print()
    console.print(f"[cyan]Next:[/cyan] EVAL_GATE_FILE={out_path} pytest tests/test_eval_gate.py -v")
    return 0


if __name__ == "__main__":
    sys.exit(main())
