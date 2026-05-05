"""Offline tests — no API calls, no network. Run with: pytest tests/"""

from pathlib import Path

from src.chunker import chunk_file, _split_by_headings


def test_split_by_headings_basic():
    text = "# Title\nintro text\n\n## Section A\nbody A\n\n## Section B\nbody B\n"
    parts = _split_by_headings(text)
    titles = [t for t, _ in parts]
    assert "Title" in titles
    assert "Section A" in titles
    assert "Section B" in titles


def test_chunker_produces_chunks(tmp_path):
    md = tmp_path / "demo.md"
    md.write_text(
        "# Pods\n\nA Pod is the smallest deployable unit in Kubernetes. "
        "Pods can contain one or more containers that share storage and network.\n\n"
        "## Pod lifecycle\n\nPods follow a defined lifecycle: Pending, Running, Succeeded, Failed.\n"
    )
    chunks = list(chunk_file(md, tmp_path))
    assert len(chunks) >= 1
    assert all(c.text.strip() for c in chunks)
    assert any("Pod" in c.text for c in chunks)


def test_chunker_preserves_code_blocks(tmp_path):
    md = tmp_path / "demo.md"
    md.write_text(
        "# Example\n\nHere is a manifest:\n\n"
        "```yaml\napiVersion: v1\nkind: Pod\n```\n\nMore prose after.\n"
    )
    chunks = list(chunk_file(md, tmp_path))
    # Code block should survive intact in some chunk
    joined = "\n".join(c.text for c in chunks)
    assert "apiVersion: v1" in joined
    assert "kind: Pod" in joined
