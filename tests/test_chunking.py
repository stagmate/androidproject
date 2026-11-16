import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safeharbor.chunking import chunk_text, split_sentences  # noqa: E402

count = lambda s: len(s.split())  # noqa: E731


def test_sentences_and_bullets():
    s = split_sentences("Fever is common. Other signs include - rash - joint pain. Seek care.")
    assert s == ["Fever is common.", "Other signs include", "rash", "joint pain.", "Seek care."]


def test_budget_overlap_and_header():
    text = " ".join(f"Sentence number {i} is here." for i in range(30))
    chunks = chunk_text(text, count, max_tokens=25, overlap_sentences=1, header="Asthma | treatment")
    assert len(chunks) > 3
    for c in chunks:
        assert c.text.startswith("Asthma | treatment: ")
        assert count(c.text) <= 25
    # one sentence of overlap between consecutive chunks
    assert chunks[0].body.split(". ")[-1].rstrip(".") in chunks[1].body
    # every sentence covered
    joined = " ".join(c.body for c in chunks)
    assert all(f"number {i} " in joined for i in range(30))


def test_oversized_sentence_is_wrapped():
    text = "word " * 100
    chunks = chunk_text(text, count, max_tokens=30, overlap_sentences=0)
    assert all(c.n_tokens <= 30 for c in chunks) and sum(c.n_tokens for c in chunks) == 100
