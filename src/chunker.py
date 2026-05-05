"""Markdown chunking. Aware of headings so chunks stay topical, and aware of code fences
so we never split a yaml/bash example down the middle."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.config import cfg


@dataclass
class Chunk:
    """A single retrievable piece of the corpus."""
    text: str
    source: str          # file path, e.g. "concepts/workloads/pods.md"
    section: str         # nearest heading, e.g. "Pod lifecycle"
    chunk_index: int     # nth chunk from this source


# Match ATX headings of any depth.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
CODE_FENCE_RE = re.compile(r"```")


def _strip_frontmatter(text: str) -> str:
    """Drop YAML frontmatter (--- ... ---) at the top of files."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split a doc into (section_title, body) pairs based on H2+ headings.

    Everything before the first heading uses the title 'Introduction'.
    """
    parts: list[tuple[str, str]] = []
    matches = list(HEADING_RE.finditer(text))

    if not matches:
        return [("Introduction", text)]

    # Pre-heading content
    if matches[0].start() > 0:
        parts.append(("Introduction", text[: matches[0].start()].strip()))

    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            parts.append((title, body))

    return parts


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Word-aware split that doesn't break inside fenced code blocks.

    We split on paragraphs first, then accumulate paragraphs until we hit chunk_size.
    Paragraphs that are themselves too big get split by sentence as a fallback.
    """
    # Protect code fences by treating them as atomic paragraphs.
    paragraphs = _split_paragraphs_preserving_code(text)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len + 2 <= chunk_size or not current:
            current.append(para)
            current_len += para_len + 2
        else:
            chunks.append("\n\n".join(current).strip())
            # overlap: carry the tail of the previous chunk into the next
            tail = _tail_for_overlap(current, overlap)
            current = [tail, para] if tail else [para]
            current_len = len(tail) + para_len + 2

    if current:
        chunks.append("\n\n".join(current).strip())

    # Edge case: a single paragraph longer than chunk_size — sentence-split it.
    final: list[str] = []
    for c in chunks:
        if len(c) <= chunk_size * 1.5:
            final.append(c)
        else:
            final.extend(_sentence_split(c, chunk_size, overlap))
    return [c for c in final if c.strip()]


def _split_paragraphs_preserving_code(text: str) -> list[str]:
    """Split on blank lines, but keep fenced code blocks together as one paragraph."""
    out: list[str] = []
    buf: list[str] = []
    in_code = False
    for line in text.splitlines():
        if CODE_FENCE_RE.search(line):
            in_code = not in_code
            buf.append(line)
            continue
        if not in_code and line.strip() == "":
            if buf:
                out.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        out.append("\n".join(buf).strip())
    return out


def _tail_for_overlap(paragraphs: list[str], overlap: int) -> str:
    """Take the last ~overlap chars of the previous chunk to seed the next one."""
    joined = "\n\n".join(paragraphs)
    if len(joined) <= overlap:
        return joined
    return joined[-overlap:]


def _sentence_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Naive sentence splitter for oversized paragraphs."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        if cur_len + len(s) + 1 <= chunk_size or not cur:
            cur.append(s)
            cur_len += len(s) + 1
        else:
            chunks.append(" ".join(cur).strip())
            cur = [s]
            cur_len = len(s) + 1
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks


def chunk_file(path: Path, root: Path) -> Iterable[Chunk]:
    """Yield Chunks from a single markdown file."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = _strip_frontmatter(raw)
    rel_source = str(path.relative_to(root))
    counter = 0
    for section, body in _split_by_headings(raw):
        for piece in _split_text(body, cfg.chunk_size, cfg.chunk_overlap):
            yield Chunk(
                text=piece,
                source=rel_source,
                section=section,
                chunk_index=counter,
            )
            counter += 1


def chunk_corpus(root: Path) -> list[Chunk]:
    """Chunk every .md file under root."""
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*.md")):
        chunks.extend(chunk_file(path, root))
    return chunks
