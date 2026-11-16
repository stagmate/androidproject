#!/usr/bin/env python
"""Retrieval-quality benchmarks (writes results/*.md and results/*.json).

  A. Embedding model comparison         (4 CPU-friendly models, same chunking)
  B. Chunking strategy ablation         (size, overlap, contextual header, naive baseline)
  C. Retrieval mode                     (dense vs BM25 vs hybrid RRF vs hybrid + section router)

Runs on a reproducible sub-corpus (default 4,000 MedQuAD documents incl. every gold document of
the sampled queries) so the whole grid finishes on a laptop CPU. Two query variants are scored:
the canonical NIH question and a paraphrase that avoids the NIH wording.

    python scripts/benchmark.py --docs 4000 --queries 400
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safeharbor.chunking import iter_chunks, make_token_counter  # noqa: E402
from safeharbor.config import ROOT, load_config  # noqa: E402
from safeharbor.data import load_medquad  # noqa: E402
from safeharbor.embeddings import MODELS, LocalEmbedder  # noqa: E402
from safeharbor.evaluation import build_query_set, metrics, pct  # noqa: E402
from safeharbor.retriever import Retriever  # noqa: E402
from safeharbor.store import create_collection, get_client, ingest  # noqa: E402

RES = ROOT / "results"


def naive_chunks(docs, size=1000, overlap=200):
    """Baseline: fixed character windows, no header (a common framework default)."""
    for d in docs:
        t = d["text"]
        i, n = 0, 0
        while True:
            body = t[i:i + size]
            yield f"{d['doc_id']}::c{n}", body, {"doc_id": d["doc_id"], "section": d["section"],
                                                  "topic": d["topic"], "header_len": 0}
            n += 1
            if i + size >= len(t):
                break
            i += size - overlap


def evaluate(client, emb, docs, queries, chunk_cfg, modes=("dense",), tag="", qcache=None):
    count = make_token_counter(emb.tokenizer)
    if chunk_cfg.get("naive"):
        triples = list(naive_chunks(docs))
    else:
        triples = list(iter_chunks(docs, count, chunk_cfg["max_tokens"], chunk_cfg["overlap_sentences"],
                                   chunk_cfg["header"]))
    ids, texts, metas = map(list, zip(*triples))
    t0 = time.perf_counter()
    vecs = emb.embed_documents(texts)
    t_emb = time.perf_counter() - t0
    col = create_collection(client, "bench", {"space": "cosine", "max_neighbors": 32,
                                              "ef_construction": 200, "ef_search": 128})
    ingest(col, ids, texts, metas, vecs)
    retr = Retriever(col, emb, build_bm25=any(m != "dense" for m in modes))

    rows = []
    for variant in ("canonical", "paraphrase"):
        qtexts = [q[variant] for q in queries]
        key = (emb.spec.name, variant)
        if qcache is not None and key in qcache:
            qvecs = qcache[key]
        else:
            qvecs = emb.embed_queries(qtexts)
            if qcache is not None:
                qcache[key] = qvecs
        for mode in modes:
            m, route = (mode, False) if mode != "hybrid+route" else ("hybrid", True)
            ranked, lat = [], []
            for qt, qv in zip(qtexts, qvecs):
                t0 = time.perf_counter()
                hits = retr.search(qt, k=10, mode=m, route=route, qvec=qv)
                lat.append((time.perf_counter() - t0) * 1000)
                ranked.append([h.doc_id for h in hits])
            r = metrics(ranked, [q["gold"] for q in queries])
            r.update(tag=tag, mode=mode, variant=variant, chunks=len(ids),
                     avg_tokens=round(sum(m_.get("n_tokens", 0) for m_ in metas) / len(metas), 1),
                     embed_chunks_per_s=round(len(ids) / t_emb, 1),
                     p50_ms=round(pct(lat, 50), 2), p95_ms=round(pct(lat, 95), 2))
            rows.append(r)
            print(f"  {tag:<38} {mode:<13} {variant:<10} R@1={r['recall@1']:.3f} R@5={r['recall@5']:.3f} "
                  f"R@10={r['recall@10']:.3f} MRR={r['mrr@10']:.3f} p50={r['p50_ms']}ms", flush=True)
    return rows


def to_md(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        body += "| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) and c.startswith(("recall", "mrr"))
                                   else str(r[c]) for c in cols) + " |\n"
    return head + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=4000)
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--only", choices=["A", "B", "C"], action="append")
    a = ap.parse_args()
    cfg = load_config()
    RES.mkdir(exist_ok=True)

    all_docs, qa = load_medquad(cfg["paths"]["raw_dir"])
    by_id = {d["doc_id"]: d for d in all_docs}
    queries = build_query_set(qa, by_id, n=a.queries)
    gold_ids = set().union(*[q["gold"] for q in queries])
    rng = random.Random(0)
    others = [d for d in all_docs if d["doc_id"] not in gold_ids]
    rng.shuffle(others)
    docs = [by_id[g] for g in sorted(gold_ids)] + others[: max(0, a.docs - len(gold_ids))]
    print(f"corpus={len(docs)} docs, queries={len(queries)} (x2 variants)")
    client = get_client(None)
    only = set(a.only or ["A", "B", "C"])
    base_chunk = {"max_tokens": 192, "overlap_sentences": 1, "header": True}
    out = {"corpus_docs": len(docs), "queries": len(queries)}

    if "A" in only:
        print("\n== A. embedding models ==")
        rows = []
        for name in MODELS:
            emb = LocalEmbedder(name)
            rows += evaluate(client, emb, docs, queries, base_chunk, ("dense",), tag=name.split("/")[-1])
        out["A"] = rows
        (RES / "embedding_benchmark.md").write_text(
            f"# A. Embedding models (dense only, 192-token header chunks, {len(docs)} docs, "
            f"{len(queries)} queries)\n\n" + to_md(rows, ["tag", "variant", "recall@1", "recall@5", "recall@10",
                                                          "mrr@10", "embed_chunks_per_s", "p50_ms"]))

    best = cfg["embedding"]["model"]
    emb = LocalEmbedder(best)
    qcache: dict = {}
    if "B" in only:
        print(f"\n== B. chunking ({best}) ==")
        grid = [
            ("naive 1000-char / 200 overlap, no header", {"naive": True}),
            ("96 tok, overlap 1, header", {"max_tokens": 96, "overlap_sentences": 1, "header": True}),
            ("192 tok, overlap 0, header", {"max_tokens": 192, "overlap_sentences": 0, "header": True}),
            ("192 tok, overlap 1, NO header", {"max_tokens": 192, "overlap_sentences": 1, "header": False}),
            ("192 tok, overlap 1, header", base_chunk),
            ("384 tok, overlap 1, header", {"max_tokens": 384, "overlap_sentences": 1, "header": True}),
        ]
        rows = []
        for tag, cc in grid:
            rows += evaluate(client, emb, docs, queries, cc, ("dense",), tag=tag, qcache=qcache)
        out["B"] = rows
        (RES / "chunking_benchmark.md").write_text(
            f"# B. Chunking ablation ({best}, dense)\n\n" + to_md(
                rows, ["tag", "variant", "chunks", "avg_tokens", "recall@1", "recall@5", "recall@10", "mrr@10"]))

    if "C" in only:
        print(f"\n== C. retrieval modes ({best}) ==")
        rows = evaluate(client, emb, docs, queries, base_chunk, ("dense", "bm25", "hybrid", "hybrid+route"),
                        tag="192 tok header", qcache=qcache)
        out["C"] = rows
        (RES / "retrieval_modes.md").write_text(
            f"# C. Retrieval modes ({best}, 192-token header chunks)\n\n" + to_md(
                rows, ["mode", "variant", "recall@1", "recall@5", "recall@10", "mrr@10", "p50_ms", "p95_ms"]))

    prev = {}
    p = RES / "benchmark.json"
    if p.exists():
        prev = json.loads(p.read_text())
    prev.update(out)
    p.write_text(json.dumps(prev, indent=2, default=list))


if __name__ == "__main__":
    main()
