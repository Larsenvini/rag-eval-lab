"""Rule-based / deterministic checks on RAG answers.

These complement the LLM-as-judge eval (which measures quality) by
measuring *contract compliance*. They're free to run, fully deterministic,
and catch a different class of bugs than the judge.

What we check:
  - Citation presence: every answer should cite at least one chunk
  - Refusal rate: track how often the model says "I don't have enough info"
  - Answer length: catch suspiciously short answers on substantive questions
  - Forbidden patterns: catch markdown headings, "as an AI" disclaimers, etc.

Usage:
    python -m scripts.code_checks evals/results/run_<ts>.json
    # prints a summary + writes evals/results/code_checks_<ts>.json

The output is a separate dimension from the LLM judge scores. The eval
gate uses it as another set of thresholds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


# ─── Patterns ─────────────────────────────────────────────────────────────

# Citation pattern: [1], [1, 2], [1,2,3], with or without spaces
CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")

# Refusal phrases — the strict-grounding system prompt makes the model say this
REFUSAL_PATTERNS = [
    re.compile(r"\bdon't have enough information\b", re.IGNORECASE),
    re.compile(r"\bnot enough information in the docs\b", re.IGNORECASE),
    re.compile(r"\bcannot answer\b", re.IGNORECASE),
    re.compile(r"\bI'm not able to\b", re.IGNORECASE),
]

# Patterns the answer should never contain (we want plain prose, no scaffolding)
FORBIDDEN_PATTERNS = {
    "as_an_ai":     re.compile(r"\bas an AI\b", re.IGNORECASE),
    "markdown_h1":  re.compile(r"^#\s+", re.MULTILINE),
    "markdown_h2":  re.compile(r"^##\s+", re.MULTILINE),
}

SUBSTANTIVE_MIN_WORDS = 25  # below this on a non-refusal answer = suspicious


# ─── Checks ───────────────────────────────────────────────────────────────


def is_refusal(text: str) -> bool:
    return any(p.search(text) for p in REFUSAL_PATTERNS)


def has_citation(text: str) -> bool:
    return bool(CITATION_RE.search(text))


def word_count(text: str) -> int:
    return len(text.split())


def forbidden_patterns_found(text: str) -> list[str]:
    return [name for name, p in FORBIDDEN_PATTERNS.items() if p.search(text)]


def check_one(result: dict) -> dict:
    """Run all checks on a single eval result. Returns per-check booleans + summary."""
    answer = result.get("answer", "") or ""
    refusal = is_refusal(answer)

    checks = {
        "is_refusal":         refusal,
        "has_citation":       has_citation(answer),
        "word_count":         word_count(answer),
        "suspiciously_short": (not refusal) and word_count(answer) < SUBSTANTIVE_MIN_WORDS,
        "forbidden_found":    forbidden_patterns_found(answer),
    }
    # Roll up into a single "passes all rules" flag
    checks["all_pass"] = (
        (checks["has_citation"] or refusal)        # refusals don't need citations
        and not checks["suspiciously_short"]
        and not checks["forbidden_found"]
    )
    return checks


def summarize(results: list[dict]) -> dict[str, Any]:
    """Aggregate stats across all results."""
    n = len(results)
    if n == 0:
        return {"n": 0}

    per_result = [check_one(r) for r in results]

    refusals = sum(1 for c in per_result if c["is_refusal"])
    no_citation = sum(1 for c in per_result if not c["has_citation"] and not c["is_refusal"])
    too_short = sum(1 for c in per_result if c["suspiciously_short"])
    has_forbidden = sum(1 for c in per_result if c["forbidden_found"])
    all_pass = sum(1 for c in per_result if c["all_pass"])

    word_counts = [c["word_count"] for c in per_result if not c["is_refusal"]]
    median_words = sorted(word_counts)[len(word_counts) // 2] if word_counts else 0

    return {
        "n": n,
        "refusals": refusals,
        "refusal_rate": round(refusals / n, 3),
        "missing_citation": no_citation,
        "suspiciously_short": too_short,
        "has_forbidden_patterns": has_forbidden,
        "all_checks_pass": all_pass,
        "pass_rate": round(all_pass / n, 3),
        "median_word_count_non_refusal": median_words,
        "per_result": [
            {"id": r.get("id"), **c}
            for r, c in zip(results, per_result)
        ],
    }


# ─── Entry point ──────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic checks on RAG answers")
    parser.add_argument("run_file", help="Path to a run_*.json (or scored_*.json — we just read .answer)")
    args = parser.parse_args()

    console = Console()
    path = Path(args.run_file)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    summary = summarize(results)

    # Pretty print
    n = summary["n"]
    table = Table(title=f"Code-check Summary  ({n} answers)")
    table.add_column("Check")
    table.add_column("Count", justify="right")
    table.add_column("Rate", justify="right")
    table.add_row("Refusals",           str(summary["refusals"]),           f"{summary['refusal_rate']:.1%}")
    table.add_row("Missing citation",   str(summary["missing_citation"]),   f"{summary['missing_citation']/n:.1%}")
    table.add_row("Suspiciously short", str(summary["suspiciously_short"]), f"{summary['suspiciously_short']/n:.1%}")
    table.add_row("Forbidden patterns", str(summary["has_forbidden_patterns"]), f"{summary['has_forbidden_patterns']/n:.1%}")
    table.add_row("All checks pass",    str(summary["all_checks_pass"]),    f"{summary['pass_rate']:.1%}")
    table.add_row("Median words (non-refusal)", str(summary["median_word_count_non_refusal"]), "—")
    console.print(table)

    # Save
    out_path = path.parent / f"code_checks_{path.stem.split('_', 1)[-1]}.json"
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"\n[green]✓ Saved → {out_path}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
