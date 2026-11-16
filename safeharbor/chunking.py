"""Structure-aware, tokenizer-exact chunking.

Design choices (benchmarked in results/chunking_benchmark.md):

* **Sentence-packing**: chunks are built from whole sentences, never cut mid-sentence, so each
  embedding encodes complete clinical statements ("Do not take X with Y").
* **Token budget in the embedding model's own tokenizer**: MiniLM silently truncates at 256
  word-pieces and bge/e5 at 512. Counting real tokens guarantees nothing is dropped.
* **Sentence overlap** instead of character overlap, so the boundary sentence is repeated whole.
* **Contextual header**: every chunk is prefixed with ``"<Topic> | <Section>:"``. A chunk that
  says "It is inherited in an autosomal recessive pattern" is useless unless the embedding
  knows *what* "it" is.
* **List awareness**: bullet runs ("- fever - rash - ...") common in NIH text are split on the
  bullet so a long list cannot form one giant "sentence".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")
_BULLET_SPLIT = re.compile(r"\s+(?:-|•|\*)\s+")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out: list[str] = []
    for sent in _SENT_SPLIT.split(text):
        parts = _BULLET_SPLIT.split(sent)
        out.extend(p.strip() for p in parts if p.strip())
    return out


@dataclass
class Chunk:
    text: str          # what gets embedded (with header)
    body: str          # chunk text without header (what the agent cites)
    index: int
    n_tokens: int


def chunk_text(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 192,
    overlap_sentences: int = 1,
    header: str | None = None,
) -> list[Chunk]:
    sents = split_sentences(text)
    if not sents:
        return []
    head = f"{header}: " if header else ""
    budget = max(16, max_tokens - (count_tokens(head) if head else 0))

    # Hard-wrap any single sentence that alone exceeds the budget (rare: long tables/lists).
    fitted: list[str] = []
    for s in sents:
        if count_tokens(s) <= budget:
            fitted.append(s)
            continue
        words, cur = s.split(), []
        for w in words:
            if cur and count_tokens(" ".join(cur + [w])) > budget:
                fitted.append(" ".join(cur))
                cur = []
            cur.append(w)
        if cur:
            fitted.append(" ".join(cur))

    lens = [count_tokens(s) for s in fitted]
    chunks: list[Chunk] = []
    i = 0
    while i < len(fitted):
        j, total = i, 0
        while j < len(fitted) and (total + lens[j] <= budget or j == i):
            total += lens[j]
            j += 1
        body = " ".join(fitted[i:j])
        chunks.append(Chunk(text=head + body, body=body, index=len(chunks), n_tokens=total))
        if j >= len(fitted):
            break
        i = max(i + 1, j - overlap_sentences)
    return chunks


def make_token_counter(tokenizer) -> Callable[[str], int]:
    def count(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))
    return count


def iter_chunks(docs: Iterable[dict], count_tokens, max_tokens, overlap_sentences, use_header=True):
    """Yield (chunk_id, chunk_text, metadata) for a stream of normalised documents."""
    for d in docs:
        header = f"{d['topic']} | {d['section']}" if use_header else None
        for c in chunk_text(d["text"], count_tokens, max_tokens, overlap_sentences, header):
            meta = {k: v for k, v in d.items() if k not in ("text",) and isinstance(v, (str, int, float, bool))}
            meta.update(chunk_index=c.index, n_tokens=c.n_tokens, header_len=len(c.text) - len(c.body))
            yield f"{d['doc_id']}::c{c.index}", c.text, meta
