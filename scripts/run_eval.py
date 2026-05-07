"""Agent 2: Eval Runner
Runs every question in golden_set.json through the RAG pipeline and saves timestamped results.

Usage:
    python -m scripts.run_eval                                  # run all questions
    python -m scripts.run_eval --topic pods                     # filter by topic
    python -m scripts.run_eval --type how-to                    # filter by type
    python -m scripts.run_eval --input evals/custom.json        # custom input file
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import track

from src.config import assert_ready, cfg
from src.generator import Generator
from src.retriever import Retriever

GOLDEN_SET_PATH = Path("evals/golden_set.json")
RESULTS_DIR = Path("evals/results")


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent 2: Eval Runner")
    parser.add_argument("--input", default=str(GOLDEN_SET_PATH), help="Golden set JSON path")
    parser.add_argument("--topic", help="Filter by topic")
    parser.add_argument("--type", dest="qtype", help="Filter by question type")
    args = parser.parse_args()

    console = Console()
    assert_ready()

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_path}[/red]")
        console.print("  Run [bold]python -m scripts.generate_questions --generate --fill-gaps[/bold] first,")
        console.print("  or add your questions manually to evals/golden_set.json")
        return 1

    data = json.loads(input_path.read_text(encoding="utf-8"))
    questions: list[dict] = data["questions"]

    if not questions:
        console.print("[yellow]golden_set.json has no questions yet.[/yellow]")
        console.print("  Run [bold]python -m scripts.generate_questions --generate --fill-gaps[/bold]")
        console.print("  or add questions manually (see evals/golden_set.json for the schema).")
        return 0

    if args.topic:
        questions = [q for q in questions if q.get("topic") == args.topic]
    if args.qtype:
        questions = [q for q in questions if q.get("type") == args.qtype]

    if not questions:
        console.print("[yellow]No questions match the given filter.[/yellow]")
        return 0

    console.print(f"[cyan]Running {len(questions)} question(s) through RAG pipeline…[/cyan]")
    console.print(f"  model={cfg.generation_model}  top_k={cfg.top_k}  temp={cfg.temperature}")

    retriever = Retriever()
    generator = Generator()
    results: list[dict] = []

    for q in track(questions, description="Querying RAG…"):
        start = time.perf_counter()
        hits = retriever.retrieve(q["question"])
        answer = generator.generate(q["question"], hits)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        results.append({
            **q,
            "answer": answer.text,
            "contexts": [
                {
                    "source": h.source,
                    "section": h.section,
                    "score": round(h.score, 4),
                    "text": h.text,
                }
                for h in answer.contexts
            ],
            "latency_ms": elapsed_ms,
        })

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{run_id}.json"
    output_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "top_k": cfg.top_k,
                    "generation_model": cfg.generation_model,
                    "embedding_model": cfg.embedding_model,
                    "chunk_size": cfg.chunk_size,
                    "chunk_overlap": cfg.chunk_overlap,
                    "temperature": cfg.temperature,
                },
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    avg_ms = sum(r["latency_ms"] for r in results) / len(results)
    console.print(f"\n[green]OK {len(results)} results saved -> {output_path}[/green]")
    console.print(f"  Avg latency: {avg_ms:.0f} ms/question")
    console.print(f"\n[cyan]Next:[/cyan] python -m scripts.score_eval {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
