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
    "factual-error": "Answer contradicts the retrieved contexts",
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

Score the RAG System Answer on three dimensions. For each, work through
the decision tree top to bottom and pick the FIRST score whose criteria
all match. Do not skip levels. Do not average. Pick exactly one.

═══════════════════════════════════════════════════════════════════════
1. FAITHFULNESS — Are the answer's claims grounded in the retrieved contexts?
═══════════════════════════════════════════════════════════════════════
Step 1: Does the answer contain ANY claim that is contradicted by the
        contexts, or contain commands/YAML keys/field names that do
        NOT appear in any context?
        YES → score is 0.0 to 0.3. Go to step 1a.
        NO  → go to step 2.

  Step 1a: Is the contradicted/invented content the primary substance
           of the answer (i.e. removing it leaves nothing)?
           YES → 0.0–0.1.   NO → 0.2–0.3.

Step 2: Does every factual claim in the answer have direct support in
        the contexts (not paraphrase, not inference — actual support)?
        YES → go to step 3.
        NO  → score is 0.4 to 0.6. The ratio of supported : unsupported
              claims determines where in that range (mostly supported = 0.6,
              half = 0.5, mostly unsupported but no contradictions = 0.4).

Step 3: Does the answer add NO information beyond what's in the contexts
        (no extra "common knowledge" beyond what was retrieved)?
        YES → 0.9–1.0. Reserve 1.0 for answers that quote/paraphrase
              chunks with zero added scaffolding.
        NO  → 0.7–0.8 if the added content is uncontroversial K8s
              common knowledge; 0.4–0.6 if it's substantive and unsupported.

═══════════════════════════════════════════════════════════════════════
2. ANSWER_RELEVANCE — Does it address what was asked?
═══════════════════════════════════════════════════════════════════════
Step 1: Does the answer refuse ("I don't have enough information") AND
        the contexts actually do contain enough information?
        YES → 0.0.

Step 2: Does the answer refuse AND the contexts genuinely lack the info?
        YES → 1.0 (correct refusal).

Step 3: Does the answer talk about a different topic than the question?
        YES → 0.0–0.2 depending on tangential relevance.

Step 4: Does the answer address the main question but ignore an explicit
        secondary part (e.g. "X and why" — answers X, ignores "why")?
        YES → 0.5–0.7. Closer to 0.7 if X is answered well.

Step 5: Does the answer address every part of the question?
        Direct and complete → 0.9–1.0
        Direct but missing one detail → 0.8–0.9

═══════════════════════════════════════════════════════════════════════
3. GROUND_TRUTH_SIMILARITY — How much of the expected answer's KEY FACTS appear?
═══════════════════════════════════════════════════════════════════════
Count the key facts in the ground truth (typically 3-6 substantive claims).
Score = (key facts present in the answer, regardless of phrasing) / (total key facts).

0.0–0.2  → almost none present, or contradicted
0.3–0.4  → minority present
0.5–0.6  → roughly half
0.7–0.8  → most present, one or two absent
0.9–1.0  → all key facts present (1.0 only if NOTHING substantive is missing)

DO NOT score on phrasing similarity. The answer can be worded entirely
differently from the ground truth; only the factual content matters.

═══════════════════════════════════════════════════════════════════════
4. FAILURE_MODE — Pick exactly one. This describes the ROOT CAUSE.
═══════════════════════════════════════════════════════════════════════
Walk this decision tree. Stop at the first match.

1. Are all three scores >= 0.85? → "none"
2. Does the answer contain factual content NOT in the contexts (claims, commands,
   YAML keys, field names that aren't there)? → "hallucination"
3. Does the answer contradict the contexts? → "factual-error"
4. Is the answer grounded and on-topic, but missing >=1 key fact from the
   ground truth? → "answer-incomplete"
5. Did the model refuse despite contexts containing the answer? → "synthesis-fail"
6. Do the retrieved contexts genuinely not contain the answer? → "retrieval-miss"
7. Is the question outside the corpus's domain? → "out-of-scope"
8. Does the answer assume Kubernetes knowledge not derivable from the
   contexts (e.g. uses jargon a learner couldn't decode from what was given)?
   → "assumes-prior"

═══════════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════════
Return ONLY a JSON object:
{{
  "faithfulness": <float 0.0-1.0>,
  "answer_relevance": <float 0.0-1.0>,
  "ground_truth_similarity": <float 0.0-1.0>,
  "failure_mode": "<one of: none, hallucination, factual-error, answer-incomplete, synthesis-fail, retrieval-miss, out-of-scope, assumes-prior>",
  "reasoning": "<2-3 sentences citing the specific decision-tree branches that led to each score>"
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
        seed=42,
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
