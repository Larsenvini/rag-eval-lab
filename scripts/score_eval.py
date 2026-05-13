"""Agent 3: Judge
Scores a run results file using GPT-4o-as-judge. Tags failure modes.

Usage:
    python -m scripts.score_eval evals/results/run_20240101_120000.json
    python -m scripts.score_eval evals/results/run_20240101_120000.json --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.progress import track
from rich.table import Table

from src.config import assert_ready, cfg

FAILURE_MODES = {
    "none": "Answer is correct and reasonably complete",
    "retrieval-miss": "Contexts don't contain the needed info (retrieval failed, not model's fault)",
    "hallucination": "Answer includes facts not in the retrieved contexts",
    "synthesis-fail": "Contexts had the answer but model failed to combine them correctly",
    "assumes-prior": "Answer assumes K8s knowledge not derivable from the contexts",
    "out-of-scope": "Question not answerable from the K8s docs corpus",
    "answer-incomplete": "Answer is grounded and relevant but shallower than the ground truth — key points missing",
}

# Canonical tag normalization: maps judge improvisation back to the defined set.
_TAG_ALIASES: dict[str, str] = {
    "ground-truth-similarity": "answer-incomplete",
    "ground_truth_similarity": "answer-incomplete",
    "gt-similarity": "answer-incomplete",
    "gt_similarity": "answer-incomplete",
    "missing-detail": "answer-incomplete",
    "incomplete": "answer-incomplete",
    "shallow": "answer-incomplete",
    "partial": "answer-incomplete",
    "partial-answer": "answer-incomplete",
}


def normalize_tag(tag: str) -> str:
    tag = tag.strip().lower()
    if tag in FAILURE_MODES:
        return tag
    if tag in _TAG_ALIASES:
        return _TAG_ALIASES[tag]
    return "answer-incomplete"  # safe fallback for any other improvisation

JUDGE_PROMPT = """\
You are a strict evaluator for a Kubernetes documentation RAG system.

QUESTION:
{question}

EXPECTED ANSWER (Ground Truth):
{ground_truth}

RETRIEVED CONTEXTS (what the RAG system was given):
{contexts}

RAG SYSTEM ANSWER:
{answer}

Score the RAG System Answer on these dimensions. Use FINE-GRAINED values — the full
range 0.0 to 1.0 in 0.1 increments is valid. Do not round everything to 0.0, 0.5,
or 1.0. Reserve 1.0 for truly complete answers and 0.0 for total failures.

1. FAITHFULNESS (0.0-1.0): Are the answer's claims traceable to the retrieved contexts?
   - 0.9-1.0 = Every claim grounded in the contexts
   - 0.6-0.8 = Mostly grounded; one or two minor unsupported details
   - 0.3-0.5 = Several claims unsupported or embellished beyond the contexts
   - 0.0-0.2 = Answer largely fabricates information not present in contexts

2. ANSWER_RELEVANCE (0.0-1.0): Does the answer address what the question actually asked?
   - 0.9-1.0 = Directly and completely answers every part of the question
   - 0.6-0.8 = Addresses the main question but misses secondary aspects
   - 0.3-0.5 = Touches the topic but misses the core ask
   - 0.0-0.2 = Off-topic or refuses without justification

3. GROUND_TRUTH_SIMILARITY (0.0-1.0): How much of the ground truth's KEY FACTS appear in the answer?
   Note: wording does not need to match — score on factual coverage, not phrasing.
   - 0.9-1.0 = All key facts from the ground truth are present
   - 0.6-0.8 = Most key facts present; one or two details absent
   - 0.3-0.5 = Some key facts present; notable gaps
   - 0.0-0.2 = Most key facts absent or contradicted

4. FAILURE_MODE: WHY is the answer wrong or incomplete? Pick exactly one.
   This describes the ROOT CAUSE, not which score was low.
   Do NOT return a metric name (like "ground-truth-similarity") as a failure mode.
   - "none" — Answer is correct and reasonably complete
   - "retrieval-miss" — The contexts don't contain the needed information (retrieval failed)
   - "hallucination" — Answer includes facts not supported by the retrieved contexts
   - "synthesis-fail" — Contexts had the answer but model failed to combine/present it correctly
   - "assumes-prior" — Answer assumes K8s knowledge not derivable from the provided contexts
   - "out-of-scope" — Question is genuinely not answerable from the K8s docs corpus
   - "answer-incomplete" — Answer is grounded and on-topic but shallower than needed; key points missing

Return ONLY a JSON object:
{{
  "faithfulness": <float 0.0-1.0>,
  "answer_relevance": <float 0.0-1.0>,
  "ground_truth_similarity": <float 0.0-1.0>,
  "failure_mode": "<one of the seven options above>",
  "reasoning": "<2-3 sentences explaining your scores and why you chose that failure mode>"
}}
"""


def format_contexts(contexts: list[dict]) -> str:
    if not contexts:
        return "(no contexts retrieved)"
    parts = []
    for i, c in enumerate(contexts, start=1):
        parts.append(f"[{i}] ({c.get('source', '?')} → {c.get('section', '?')})\n{c.get('text', '')}")
    return "\n\n".join(parts)


def judge_one(client: OpenAI, result: dict, model: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=result.get("question", ""),
        ground_truth=result.get("ground_truth", "N/A — no ground truth provided"),
        contexts=format_contexts(result.get("contexts", [])),
        answer=result.get("answer", ""),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content or "{}")
    data["failure_mode"] = normalize_tag(data.get("failure_mode", ""))
    return data


def print_summary(scored: list[dict], run_id: str, console: Console) -> None:
    table = Table(title=f"Eval Results — {run_id}", show_header=True)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Topic", no_wrap=True)
    table.add_column("Type", no_wrap=True)
    table.add_column("Faith.", justify="right")
    table.add_column("Rel.", justify="right")
    table.add_column("GT Sim.", justify="right")
    table.add_column("Failure Mode")

    failure_counts: dict[str, int] = {}
    for r in scored:
        fm = r.get("failure_mode", "unknown")
        failure_counts[fm] = failure_counts.get(fm, 0) + 1
        color = "green" if fm == "none" else ("yellow" if fm == "retrieval-miss" else "red")
        table.add_row(
            r.get("id", "?"),
            r.get("topic", "?"),
            r.get("type", "?"),
            f"{r.get('faithfulness', 0):.2f}",
            f"{r.get('answer_relevance', 0):.2f}",
            f"{r.get('ground_truth_similarity', 0):.2f}",
            f"[{color}]{fm}[/{color}]",
        )

    console.print(table)

    n = len(scored)
    avg = lambda key: sum(r.get(key, 0) for r in scored) / n
    console.print(
        f"\n[bold]Averages[/bold] (n={n}):  "
        f"faithfulness={avg('faithfulness'):.2f}  "
        f"relevance={avg('answer_relevance'):.2f}  "
        f"gt_similarity={avg('ground_truth_similarity'):.2f}"
    )

    console.print("\n[bold]Failure mode breakdown:[/bold]")
    for fm, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / n
        bar = "#" * max(1, int(pct / 4))
        desc = FAILURE_MODES.get(fm, "")
        console.print(f"  {fm:<20} {bar:<25} {count:>2} ({pct:4.0f}%)  [dim]{desc}[/dim]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent 3: Judge")
    parser.add_argument("results_file", help="Path to a run_*.json results file")
    parser.add_argument(
        "--model", default="gpt-4o",
        help="Judge model (default: gpt-4o; use gpt-4o-mini to cut cost)"
    )
    args = parser.parse_args()

    console = Console()
    assert_ready()

    results_path = Path(args.results_file)
    if not results_path.exists():
        console.print(f"[red]Results file not found: {results_path}[/red]")
        return 1

    run_data = json.loads(results_path.read_text(encoding="utf-8"))
    results: list[dict] = run_data["results"]

    if not results:
        console.print("[yellow]Results file is empty.[/yellow]")
        return 0

    console.print(
        f"[cyan]Scoring {len(results)} result(s) with {args.model}…[/cyan]\n"
        f"  Tip: pass --model gpt-4o-mini to reduce cost (~10x cheaper, slightly less accurate)"
    )

    client = OpenAI(api_key=cfg.openai_api_key)
    scored: list[dict] = []

    for r in track(results, description="Judging…"):
        try:
            scores = judge_one(client, r, args.model)
        except Exception as exc:
            scores = {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "ground_truth_similarity": 0.0,
                "failure_mode": "error",
                "reasoning": str(exc),
            }
        scored.append({**r, **scores})

    scored_path = results_path.parent / results_path.name.replace("run_", "scored_")
    scored_path.write_text(
        json.dumps(
            {
                **run_data,
                "judge_model": args.model,
                "scored_at": datetime.now().isoformat(),
                "results": scored,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print_summary(scored, run_data.get("run_id", "?"), console)
    console.print(f"\n[green]OK Scored results saved -> {scored_path}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
