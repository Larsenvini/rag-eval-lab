"""Download the slices of the Kubernetes documentation we want to index.

We pull from kubernetes/website on GitHub and copy specific subdirectories.
This avoids depending on a fragile scraper.

Run once:
    python -m scripts.download_corpus
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rich.console import Console

from src.config import cfg


# These are the doc paths we care about, relative to content/en/docs/ in
# kubernetes/website. We're keeping it tight: foundational concepts +
# practical "how do I..." tasks that recruiters love asking about.
INCLUDE_PATHS: list[str] = [
    "concepts/overview",
    "concepts/architecture",
    "concepts/workloads",
    "concepts/services-networking",
    "concepts/storage",
    "concepts/configuration",
    "concepts/cluster-administration",
    "tasks/configure-pod-container",
    "tasks/run-application",
    "tasks/access-application-cluster",
]

REPO_URL = "https://github.com/kubernetes/website.git"
REPO_DOCS_PREFIX = Path("content/en/docs")


def main() -> int:
    console = Console()
    console.print("[cyan]Cloning kubernetes/website (shallow)…[/cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", REPO_URL, str(tmp / "website")],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            console.print("[red]git clone failed:[/red]")
            console.print(e.stderr.decode("utf-8", errors="ignore"))
            return 1
        except FileNotFoundError:
            console.print("[red]git is not installed or not in PATH.[/red]")
            return 1

        docs_root = tmp / "website" / REPO_DOCS_PREFIX
        if not docs_root.exists():
            console.print(f"[red]Expected path not found: {docs_root}[/red]")
            return 2

        # Wipe the local raw dir so re-runs are clean
        if cfg.raw_corpus_path.exists():
            shutil.rmtree(cfg.raw_corpus_path)
        cfg.raw_corpus_path.mkdir(parents=True, exist_ok=True)

        total_files = 0
        for rel in INCLUDE_PATHS:
            src = docs_root / rel
            dst = cfg.raw_corpus_path / rel
            if not src.exists():
                console.print(f"[yellow]Skipping missing path: {rel}[/yellow]")
                continue
            shutil.copytree(src, dst)
            count = sum(1 for _ in dst.rglob("*.md"))
            total_files += count
            console.print(f"  ✓ {rel}  ([dim]{count} files[/dim])")

    console.print()
    console.print(f"[green]Downloaded {total_files} markdown files → {cfg.raw_corpus_path}[/green]")
    console.print("[cyan]Next:[/cyan] python -m scripts.ingest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
