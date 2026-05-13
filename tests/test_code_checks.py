"""Tests for scripts.code_checks — fast, deterministic, no API."""

from __future__ import annotations

from scripts.code_checks import (
    check_one,
    forbidden_patterns_found,
    has_citation,
    is_refusal,
    summarize,
    word_count,
)


# ─── is_refusal ──────────────────────────────────────────────────────────


def test_refusal_phrase_detected():
    assert is_refusal("I don't have enough information in the docs to answer that.")
    assert is_refusal("I'm not able to answer that based on the provided docs.")
    assert is_refusal("There is not enough information in the docs.")


def test_normal_answer_is_not_refusal():
    assert not is_refusal("A Pod is the smallest deployable unit in Kubernetes. [1]")


# ─── has_citation ────────────────────────────────────────────────────────


def test_citation_recognized_in_various_forms():
    assert has_citation("A Pod is small. [1]")
    assert has_citation("Two sources [1, 2]")
    assert has_citation("Three sources [1,2,3]")
    assert has_citation("End of answer [12]")


def test_no_citation_in_plain_prose():
    assert not has_citation("A Pod is the smallest deployable unit in Kubernetes.")


def test_brackets_alone_dont_count():
    assert not has_citation("Some text with [brackets] but no numbers.")


# ─── word_count ──────────────────────────────────────────────────────────


def test_word_count_basic():
    assert word_count("Hello world") == 2
    assert word_count("") == 0


# ─── forbidden_patterns ──────────────────────────────────────────────────


def test_as_an_ai_caught():
    found = forbidden_patterns_found("As an AI language model, I cannot...")
    assert "as_an_ai" in found


def test_markdown_headings_caught():
    found = forbidden_patterns_found("## Heading\nSome answer.")
    assert "markdown_h2" in found


def test_clean_answer_has_no_forbidden():
    found = forbidden_patterns_found("A Pod is the smallest deployable unit. [1]")
    assert found == []


# ─── check_one ───────────────────────────────────────────────────────────


def _result(answer: str, qid: str = "Q01") -> dict:
    return {"id": qid, "answer": answer}


def test_check_one_clean_answer_passes():
    answer = (
        "A Pod is the smallest deployable unit in Kubernetes, containing one "
        "or more containers that share network and storage. Pods are managed "
        "by higher-level controllers like Deployments. [1, 2]"
    )
    c = check_one(_result(answer))
    assert c["all_pass"] is True
    assert c["has_citation"] is True
    assert c["is_refusal"] is False
    assert c["suspiciously_short"] is False


def test_check_one_refusal_passes_without_citation():
    """Refusals are allowed to skip citation."""
    c = check_one(_result("I don't have enough information in the docs to answer that."))
    assert c["is_refusal"] is True
    assert c["has_citation"] is False
    # Refusals are valid (the contract permits them) — all_pass should be True
    # as long as no other rule is violated.
    assert c["all_pass"] is True


def test_check_one_short_answer_fails():
    c = check_one(_result("A Pod is small. [1]"))   # 4 words, has citation
    assert c["suspiciously_short"] is True
    assert c["all_pass"] is False


def test_check_one_missing_citation_fails():
    long_uncited = " ".join(["word"] * 50)
    c = check_one(_result(long_uncited))
    assert c["has_citation"] is False
    assert c["all_pass"] is False


def test_check_one_forbidden_pattern_fails():
    """Long answer, has citation, but uses forbidden phrasing."""
    answer = "As an AI, I will explain. " + " ".join(["word"] * 50) + " [1]"
    c = check_one(_result(answer))
    assert "as_an_ai" in c["forbidden_found"]
    assert c["all_pass"] is False


# ─── summarize ───────────────────────────────────────────────────────────


def test_summarize_mixed_results():
    long_ok = " ".join(["word"] * 60) + " [1]"
    short = "Too short [1]"
    refusal = "I don't have enough information in the docs to answer that."
    bad = "As an AI... " + " ".join(["word"] * 50) + " [1]"
    uncited = " ".join(["word"] * 60)

    results = [
        _result(long_ok, "Q1"),
        _result(short, "Q2"),
        _result(refusal, "Q3"),
        _result(bad, "Q4"),
        _result(uncited, "Q5"),
    ]
    s = summarize(results)
    assert s["n"] == 5
    assert s["refusals"] == 1
    assert s["missing_citation"] == 1  # Q5 — Q3 is a refusal, doesn't count
    assert s["suspiciously_short"] == 1
    assert s["has_forbidden_patterns"] == 1
    assert s["all_checks_pass"] == 2  # only Q1 and Q3 pass everything


def test_summarize_empty():
    assert summarize([])["n"] == 0
