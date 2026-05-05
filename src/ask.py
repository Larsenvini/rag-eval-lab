"""CLI entrypoint.

Usage:
    python -m src.ask "How do I expose a deployment?"
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel

from src.config import assert_ready
from src.generator import Generator
from src.retriever import Retriever


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m src.ask "your question here"', file=sys.stderr)
        return 1

    question = " ".join(sys.argv[1:])
    assert_ready()

    console = Console()

    with console.status("[cyan]Retrieving relevant chunks…", spinner="dots"):
        retriever = Retriever()
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

    sources_text = "\n".join(
        f"[{i}] {c.source} → {c.section}  [dim](dist={c.score:.3f})[/dim]"
        for i, c in enumerate(answer.contexts, start=1)
    )
    console.print(Panel.fit(sources_text, title="Sources", border_style="dim"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
