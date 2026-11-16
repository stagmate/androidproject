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
from safeharbor.data import load_medquad  # noqa: E402
from safeharbor.embeddings import LocalEmbedder  # noqa: E402
from safeharbor.evaluation import build_query_set, pct  # noqa: E402
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
    # Two query sets, both scored against exact brute-force cosine top-10:
    #  * "perturbed": in-corpus vectors + Gaussian noise. The usual synthetic ANN benchmark.
    #  * "real": the 1,000 evaluation questions (canonical + paraphrase) embedded with the
    #    production model. On MedQuAD these land between tight clusters of near-duplicate NIH
    #    boilerplate, where greedy HNSW search stalls; the perturbed set badly overestimates recall.
    qi = rng.choice(len(X), a.queries, replace=False)
    Qp = X[qi] + rng.normal(0, 0.03, size=(a.queries, X.shape[1])).astype(np.float32)
    Qp /= np.linalg.norm(Qp, axis=1, keepdims=True)
    all_docs, qa = load_medquad(cfg["paths"]["raw_dir"])
    qs = build_query_set(qa, {d["doc_id"]: d for d in all_docs}, n=500)
    emb = LocalEmbedder(cfg["embedding"]["model"])
    Qr = emb.embed_queries([q["canonical"] for q in qs] + [q["paraphrase"] for q in qs])
    del emb
    exact = {name: [{ids[j] for j in row} for row in np.argsort(-(Q @ X.T), axis=1)[:, :10]]
             for name, Q in (("perturbed", Qp), ("real", Qr))}
    exact_top1 = [ids[int(np.argmax(X @ q))] for q in Qr]

    t = []
    for q in Qr[:200]:
        t0 = time.perf_counter(); np.argpartition(-(X @ q), 10)[:10]; t.append((time.perf_counter() - t0) * 1000)
    rows = [{"M": "exact (numpy)", "ef_construction": "-", "ef_search": "-", "recall@10 perturbed": 1.0,
             "recall@10 real": 1.0, "top1 lost (real)": 0, "p50_ms": round(pct(t, 50), 2),
             "p95_ms": round(pct(t, 95), 2), "build_s": 0, "disk_mb": "-"}]

    # NOTE: in Chroma 1.5, collection.modify(ef_search=...) is persisted but only applies once the DB
    # is reopened in a new process; it does not change an index already loaded in memory. To keep
    # the sweep in one process, each (M, ef_construction, ef_search) point gets its own fresh build.
    for M, efc in [(16, 200), (32, 200), (48, 400), (64, 400)]:
        for ef in (32, 64, 128, 256, 512, 1024):
            tmp = tempfile.mkdtemp(prefix="hnsw_")
            client = get_client(tmp)
            col = create_collection(client, "hnsw-bench", {"space": "cosine", "max_neighbors": M,
                                                           "ef_construction": efc, "ef_search": ef})
            t0 = time.perf_counter()
            ingest(col, ids, docs, metas, X)
            col.count()
            build = time.perf_counter() - t0
            size = dir_size_mb(tmp)
            col.query(query_embeddings=Qr[:5].tolist(), n_results=10)  # warm
            row = {"M": M, "ef_construction": efc, "ef_search": ef}
            for name, Q in (("perturbed", Qp), ("real", Qr)):
                lat, rec, lost = [], [], 0
                for i, q in enumerate(Q):
                    t0 = time.perf_counter()
                    r = col.query(query_embeddings=[q.tolist()], n_results=10, include=[])
                    lat.append((time.perf_counter() - t0) * 1000)
                    rec.append(len(set(r["ids"][0]) & exact[name][i]) / 10)
                    if name == "real" and r["ids"][0][0] != exact_top1[i]:
                        lost += 1
                row[f"recall@10 {name}"] = round(float(np.mean(rec)), 4)
                if name == "real":
                    row.update({"top1 lost (real)": lost, "p50_ms": round(pct(lat, 50), 2),
                                "p95_ms": round(pct(lat, 95), 2)})
            row.update(build_s=round(build, 1), disk_mb=round(size, 1))
            rows.append(row)
            print(row, flush=True)
            del col, client
            shutil.rmtree(tmp, ignore_errors=True)

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    cols = ["M", "ef_construction", "ef_search", "recall@10 perturbed", "recall@10 real", "top1 lost (real)",
            "p50_ms", "p95_ms", "build_s", "disk_mb"]
    md = (f"# D. HNSW tuning ({X.shape[0]} vectors x {X.shape[1]} dims)\n\n"
          "ANN recall@10 against exact brute-force cosine search, for two query sets: "
          f"{a.queries} perturbed in-corpus vectors, and the {len(Qr)} real evaluation questions. "
          "Latency is per single real query. `top1 lost` = real queries whose exact nearest neighbour "
          "is not returned first.\n\n"
          "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    md += "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)
    (res / "hnsw_benchmark.md").write_text(md)
    (res / "hnsw_benchmark.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
