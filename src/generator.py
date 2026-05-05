"""Generator: chunks + question → grounded answer with citations.

The system prompt is intentionally strict about grounding — this is what
the eval framework will measure in Week 4.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from src.config import cfg
from src.store import Hit


SYSTEM_PROMPT = """You are a Kubernetes documentation assistant.

Answer the user's question using ONLY the provided documentation excerpts.

Rules:
- If the answer is in the excerpts, write a concise, accurate answer (3-6 sentences).
- After your answer, list which excerpt numbers you used like: [1, 3].
- If the excerpts do NOT contain the answer, reply exactly: "I don't have enough information in the docs to answer that."
- Do NOT use any outside knowledge of Kubernetes.
- Do NOT invent commands, fields, or YAML keys.
- If asked for an example, only provide one if a similar example exists in the excerpts.
"""


@dataclass
class Answer:
    text: str
    contexts: list[Hit]      # the chunks that were actually shown to the model

    def format(self) -> str:
        """Pretty CLI output."""
        lines = [self.text, "", "─── Sources ───"]
        for i, c in enumerate(self.contexts, start=1):
            lines.append(f"[{i}] {c.source} → {c.section}  (score={c.score:.3f})")
        return "\n".join(lines)


class Generator:
    def __init__(self) -> None:
        self._openai = OpenAI(api_key=cfg.openai_api_key)

    def generate(self, question: str, contexts: list[Hit]) -> Answer:
        if not contexts:
            return Answer(
                text="I don't have enough information in the docs to answer that.",
                contexts=[],
            )

        # Build the user message
        excerpt_block = "\n\n".join(
            f"[{i}] (from {c.source} → {c.section})\n{c.text}"
            for i, c in enumerate(contexts, start=1)
        )
        user_msg = f"DOCUMENTATION EXCERPTS:\n{excerpt_block}\n\nQUESTION: {question}"

        response = self._openai.chat.completions.create(
            model=cfg.generation_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=cfg.temperature,
            max_tokens=cfg.max_answer_tokens,
        )
        text = response.choices[0].message.content or ""
        return Answer(text=text.strip(), contexts=contexts)
