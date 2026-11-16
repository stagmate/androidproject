#!/usr/bin/env python
"""D. HNSW tuning on the full production index.

For every (max_neighbors=M, ef_construction) build and every ef_search we measure:
  * ANN recall@10 against exact brute-force cosine search over the same vectors
  * query latency p50 / p95 (single query, warm)
  * build time and on-disk size
Vectors are read back from the shipped Chroma DB so the benchmark sees exactly what Yankee AI sees.

    python scripts/benchmark_hnsw.py --queries 300
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safeharbor.config import ROOT, load_config  # noqa: E402
from safeharbor.evaluation import pct  # noqa: E402
from safeharbor.store import create_collection, dir_size_mb, get_all, get_client, ingest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=300)
    a = ap.parse_args()
    cfg = load_config()
    src = get_client(cfg["paths"]["chroma_dir"]).get_collection("medical_kb")
    got = get_all(src, ["embeddings", "documents", "metadatas"])
    ids, X = got["ids"], np.asarray(got["embeddings"], dtype=np.float32)
    docs, metas = got["documents"], got["metadatas"]
    print(f"vectors: {X.shape}")

    rng = np.random.default_rng(0)
    # Queries: real question embeddings would be ideal, but to isolate the *index* error we use
    # held-in vectors perturbed with noise (so the true neighbour is not trivially itself).
    qi = rng.choice(len(X), a.queries, replace=False)
    Q = X[qi] + rng.normal(0, 0.03, size=(a.queries, X.shape[1])).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    exact = np.argsort(-(Q @ X.T), axis=1)[:, :10]
    exact_ids = [{ids[j] for j in row} for row in exact]

    # Brute force latency for reference
    t = []
    for q in Q[:100]:
        t0 = time.perf_counter(); np.argpartition(-(X @ q), 10)[:10]; t.append((time.perf_counter() - t0) * 1000)
    rows = [{"M": "exact", "ef_construction": "-", "ef_search": "-", "recall@10": 1.0,
             "p50_ms": round(pct(t, 50), 2), "p95_ms": round(pct(t, 95), 2), "build_s": 0, "disk_mb": "-"}]

    # NOTE: in Chroma 1.5, collection.modify(ef_search=...) is persisted but only applies once the DB
    # is reopened in a new process; it does not change an index already loaded in memory. To keep
    # the sweep in one process, each (M, ef_construction, ef_search) point gets its own fresh build.
    for M, efc in [(8, 100), (16, 100), (16, 200), (32, 200), (48, 400)]:
        for ef in (10, 32, 64, 128, 256):
            tmp = tempfile.mkdtemp(prefix="hnsw_")
            client = get_client(tmp)
            col = create_collection(client, "hnsw-bench", {"space": "cosine", "max_neighbors": M,
                                                           "ef_construction": efc, "ef_search": ef})
            t0 = time.perf_counter()
            ingest(col, ids, docs, metas, X)
            col.count()
            build = time.perf_counter() - t0
            size = dir_size_mb(tmp)
            col.query(query_embeddings=Q[:5].tolist(), n_results=10)  # warm
            lat, rec = [], []
            for q, ex in zip(Q, exact_ids):
                t0 = time.perf_counter()
                r = col.query(query_embeddings=[q.tolist()], n_results=10, include=[])
                lat.append((time.perf_counter() - t0) * 1000)
                rec.append(len(set(r["ids"][0]) & ex) / 10)
            row = {"M": M, "ef_construction": efc, "ef_search": ef, "recall@10": round(float(np.mean(rec)), 4),
                   "p50_ms": round(pct(lat, 50), 2), "p95_ms": round(pct(lat, 95), 2),
                   "build_s": round(build, 1), "disk_mb": round(size, 1)}
            rows.append(row)
            print(row, flush=True)
            del col, client
            shutil.rmtree(tmp, ignore_errors=True)

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    cols = ["M", "ef_construction", "ef_search", "recall@10", "p50_ms", "p95_ms", "build_s", "disk_mb"]
    md = (f"# D. HNSW tuning ({X.shape[0]} vectors x {X.shape[1]} dims, {a.queries} queries)\n\n"
          "ANN recall@10 is measured against exact brute-force cosine search.\n\n"
          "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    md += "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)
    (res / "hnsw_benchmark.md").write_text(md)
    (res / "hnsw_benchmark.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
