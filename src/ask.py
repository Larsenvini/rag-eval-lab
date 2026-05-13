"""CLI entrypoint.

Usage:
    python -m src.ask "How do I expose a deployment?"
    python -m src.ask --retriever hybrid "How do I expose a deployment?"
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel

from src.config import assert_ready, cfg
from src.generator import Generator
from src.hybrid_retriever import build_retriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the K8s docs RAG.")
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument(
        "--retriever",
        choices=["dense", "hybrid", "hybrid+rerank", "hybrid+rerank-large"],
        default=cfg.default_retriever,
        help="Retriever to use (default: %(default)s)",
    )
    args = parser.parse_args()

    question = " ".join(args.question)
    assert_ready()
    console = Console()

    with console.status(f"[cyan]Retrieving via {args.retriever}…", spinner="dots"):
        retriever = build_retriever(args.retriever)
        # Both retrievers expose .store with a .count property
        if retriever.store.count == 0:
            console.print(
                "[red]Vector store is empty.[/red] "
                "Run [bold]python -m scripts.ingest[/bold] first."
            )
            return 2
        contexts = retriever.retrieve(question)

    with console.status("[cyan]Generating answer…", spinner="dots"):
        generator = Generator()
        answer = generator.generate(question, contexts)

    console.print(Panel.fit(question, title="Question", border_style="dim"))
    console.print(Panel.fit(answer.text, title="Answer", border_style="cyan"))

    score_label = {
        "dense": "dist",
        "hybrid": "rrf",
        "hybrid+rerank": "ce",          # cross-encoder (MiniLM)
        "hybrid+rerank-large": "bge",   # BGE cross-encoder
    }[args.retriever]
    sources_text = "\n".join(
        f"[{i}] {c.source} → {c.section}  [dim]({score_label}={c.score:.3f})[/dim]"
        for i, c in enumerate(answer.contexts, start=1)
    )
    console.print(Panel.fit(sources_text, title=f"Sources ({args.retriever})", border_style="dim"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
