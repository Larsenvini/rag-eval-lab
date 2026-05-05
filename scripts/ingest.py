"""Chunk the downloaded corpus, embed it, and load it into ChromaDB.

Run once after download_corpus.py:
    python -m scripts.ingest

It's safe to re-run: the collection is reset before re-embedding.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

from src.chunker import chunk_corpus
from src.config import assert_ready, cfg
from src.store import VectorStore


def main() -> int:
    console = Console()
    assert_ready()

    if not cfg.raw_corpus_path.exists() or not any(cfg.raw_corpus_path.rglob("*.md")):
        console.print(
            "[red]No corpus found at[/red] "
            f"[bold]{cfg.raw_corpus_path}[/bold]. "
            "Run [bold]python -m scripts.download_corpus[/bold] first."
        )
        return 1

    console.print(f"[cyan]Chunking corpus at[/cyan] {cfg.raw_corpus_path}…")
    chunks = chunk_corpus(cfg.raw_corpus_path)
    console.print(f"  Got [bold]{len(chunks)}[/bold] chunks "
                  f"(chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap})")

    if not chunks:
        console.print("[red]No chunks produced — something's off with the corpus.[/red]")
        return 2

    # A peek at chunk-size distribution so we can sanity-check Week 3 sweeps.
    sizes = [len(c.text) for c in chunks]
    table = Table(title="Chunk size distribution", show_header=True)
    table.add_column("min")
    table.add_column("p50")
    table.add_column("p90")
    table.add_column("max")
    sizes_sorted = sorted(sizes)
    table.add_row(
        str(sizes_sorted[0]),
        str(sizes_sorted[len(sizes_sorted) // 2]),
        str(sizes_sorted[int(len(sizes_sorted) * 0.9)]),
        str(sizes_sorted[-1]),
    )
    console.print(table)

    console.print("[cyan]Resetting and re-embedding…[/cyan]")
    store = VectorStore()
    store.reset()
    store.add_chunks(chunks)

    console.print(f"[green]✓ Ingested {store.count} chunks into "
                  f"'{cfg.collection_name}'[/green]")
    console.print("[cyan]Try it:[/cyan] python -m src.ask \"How does a Pod differ from a container?\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
