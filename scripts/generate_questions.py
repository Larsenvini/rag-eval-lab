"""Agent 1: Question Generator
Fills the coverage matrix (topics x types) using GPT-4o to draft question + ground truth pairs.

Usage:
    python -m scripts.generate_questions                               # show coverage matrix
    python -m scripts.generate_questions --generate --topic pods --type definition
    python -m scripts.generate_questions --generate --fill-gaps        # fill all empty cells
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.table import Table

from src.config import assert_ready, cfg

TOPICS = [
    "pods",
    "deployments",
    "services",
    "configmaps-secrets",
    "storage-volumes",
    "networking-ingress",
    "scheduling-nodes",
    "architecture",
    "out-of-scope",
]

TYPES = ["definition", "how-to", "comparison", "why-tradeoff", "edge-case"]

GOLDEN_SET_PATH = Path("evals/golden_set.json")

GENERATE_PROMPT = """\
You are helping build a golden evaluation set for a Kubernetes documentation RAG system.
The RAG system is grounded in the official Kubernetes documentation (Concepts + Tasks sections).

Generate exactly ONE "{qtype}" question about the topic "{topic}" in Kubernetes.

Question type definitions:
- definition: "What is X?" — tests basic retrieval, answer should be near copy-paste from docs
- how-to: "How do I configure/use X?" — tests procedural synthesis, answer is step-by-step
- comparison: "What is the difference between X and Y?" — tests multi-concept retrieval
- why-tradeoff: "Why would I use X over Y?" — tests reasoning over docs, models love to hallucinate here
- edge-case: "What happens when X fails/crashes/is missing?" — tests failure-mode coverage, needs multi-chunk synthesis

Requirements:
1. The question MUST be answerable from the official Kubernetes documentation
2. Write a ground truth answer of 3-5 sentences that a K8s expert would agree with
3. The ground truth must be verifiable from the official K8s docs
4. Do NOT invent YAML fields, flags, or behaviors that do not exist in K8s

Return ONLY a JSON object with exactly these keys:
{{
  "question": "<the question text>",
  "ground_truth": "<3-5 sentence accurate answer>",
  "difficulty": "easy" or "medium" or "hard",
  "notes": "<optional: why this is a good eval question, leave empty string if none>"
}}
"""


def load_golden_set() -> dict:
    if GOLDEN_SET_PATH.exists():
        return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    return {"version": "1.0", "questions": []}


def save_golden_set(data: dict) -> None:
    GOLDEN_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_SET_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_coverage(questions: list[dict]) -> set[tuple[str, str]]:
    return {(q["topic"], q["type"]) for q in questions}


def show_matrix(questions: list[dict], console: Console) -> None:
    covered = build_coverage(questions)
    table = Table(title="Coverage Matrix", show_header=True)
    table.add_column("Topic", style="bold")
    for t in TYPES:
        table.add_column(t, justify="center")

    total_cells = 0
    filled_cells = 0
    for topic in TOPICS:
        row = [topic]
        for qtype in TYPES:
            if topic == "out-of-scope" and qtype != "definition":
                row.append("[dim]-[/dim]")
            elif (topic, qtype) in covered:
                count = sum(1 for q in questions if q["topic"] == topic and q["type"] == qtype)
                row.append(f"[green]+({count})[/green]")
                filled_cells += 1
                total_cells += 1
            else:
                row.append("[red].[/red]")
                total_cells += 1
        table.add_row(*row)

    console.print(table)
    console.print(
        f"\n[bold]{filled_cells}/{total_cells}[/bold] cells filled  "
        f"[bold]{len(questions)}[/bold] total questions"
    )


def next_id(questions: list[dict]) -> str:
    nums = [int(q["id"][1:]) for q in questions if q.get("id", "").startswith("Q")]
    return f"Q{(max(nums) + 1) if nums else 1:02d}"


def generate_one(client: OpenAI, topic: str, qtype: str) -> dict:
    prompt = GENERATE_PROMPT.format(topic=topic, qtype=qtype)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content or "{}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent 1: Question Generator")
    parser.add_argument("--generate", action="store_true", help="Generate new questions")
    parser.add_argument("--topic", help="Topic to generate for (e.g. pods)")
    parser.add_argument("--type", dest="qtype", help="Question type (e.g. definition)")
    parser.add_argument("--fill-gaps", action="store_true", help="Fill all empty matrix cells")
    args = parser.parse_args()

    console = Console()
    data = load_golden_set()
    questions: list[dict] = data["questions"]

    if not args.generate:
        show_matrix(questions, console)
        console.print("\n[cyan]To generate:[/cyan] add --generate --fill-gaps  "
                      "or --generate --topic pods --type definition")
        return 0

    assert_ready()
    client = OpenAI(api_key=cfg.openai_api_key)
    covered = build_coverage(questions)

    targets: list[tuple[str, str]] = []
    if args.fill_gaps:
        for topic in TOPICS:
            for qtype in TYPES:
                if topic == "out-of-scope" and qtype != "definition":
                    continue
                if (topic, qtype) not in covered:
                    targets.append((topic, qtype))
    elif args.topic and args.qtype:
        targets = [(args.topic, args.qtype)]
    else:
        console.print("[red]Specify --fill-gaps  or both --topic and --type[/red]")
        return 1

    console.print(f"[cyan]Generating {len(targets)} question(s) with gpt-4o…[/cyan]")
    for topic, qtype in targets:
        console.print(f"  {topic} / {qtype}…", end=" ")
        try:
            result = generate_one(client, topic, qtype)
            qid = next_id(questions)
            questions.append({
                "id": qid,
                "question": result["question"],
                "topic": topic,
                "type": qtype,
                "source": "synthetic-edited",
                "ground_truth": result["ground_truth"],
                "difficulty": result.get("difficulty", "medium"),
                "notes": result.get("notes", ""),
            })
            save_golden_set(data)
            console.print(f"[green]OK {qid}[/green]")
        except Exception as exc:
            console.print(f"[red]FAIL {exc}[/red]")

    console.print()
    show_matrix(questions, console)
    return 0


if __name__ == "__main__":
    sys.exit(main())
