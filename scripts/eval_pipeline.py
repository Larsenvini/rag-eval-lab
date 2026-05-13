"""Orchestrator: chains generate → run → score in one command.

Usage:
    python -m scripts.eval_pipeline                     # run → score (most common)
    python -m scripts.eval_pipeline --with-generate     # generate missing questions first
    python -m scripts.eval_pipeline --only-generate     # just generate, don't run eval
    python -m scripts.eval_pipeline --topic pods        # filter run to one topic
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

RESULTS_DIR = Path("evals/results")


def run_step(cmd: list[str], console: Console) -> int:
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]\n")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def latest_run_file() -> Path | None:
    files = sorted(RESULTS_DIR.glob("run_*.json"), reverse=True)
    return files[0] if files else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval Pipeline Orchestrator")
    parser.add_argument("--with-generate", action="store_true",
                        help="Generate missing questions before running eval")
    parser.add_argument("--only-generate", action="store_true",
                        help="Only generate questions, skip run and score")
    parser.add_argument("--topic", help="Limit run_eval to a single topic")
    parser.add_argument("--type", dest="qtype", help="Limit run_eval to a single question type")
    parser.add_argument("--judge-model", default="gpt-4o",
                        help="Model for score_eval judge (default: gpt-4o)")
    args = parser.parse_args()

    console = Console()
    python = sys.executable

    if args.with_generate or args.only_generate:
        console.print(Rule("[bold cyan]Step 1 — Generate Questions[/bold cyan]"))
        rc = run_step(
            [python, "-m", "scripts.generate_questions", "--generate", "--fill-gaps"],
            console,
        )
        if rc != 0:
            console.print("[red]generate_questions failed.[/red]")
            return rc
        if args.only_generate:
            return 0

    console.print(Rule("[bold cyan]Step 2 — Run Eval[/bold cyan]"))
    run_cmd = [python, "-m", "scripts.run_eval"]
    if args.topic:
        run_cmd += ["--topic", args.topic]
    if args.qtype:
        run_cmd += ["--type", args.qtype]
    rc = run_step(run_cmd, console)
    if rc != 0:
        console.print("[red]run_eval failed.[/red]")
        return rc

    run_file = latest_run_file()
    if not run_file:
        console.print("[red]No run results file found — something went wrong.[/red]")
        return 1

    console.print(Rule("[bold cyan]Step 3 — Score Results[/bold cyan]"))
    rc = run_step(
        [python, "-m", "scripts.score_eval", str(run_file), "--model", args.judge_model],
        console,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
