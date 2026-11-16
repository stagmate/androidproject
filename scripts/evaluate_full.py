#!/usr/bin/env python
"""F. End-to-end evaluation on the full shipped index + failure analysis.

Uses the exact production path (SafeHarborTools, airgapped) on all 35,811 chunks, with the same
500-query sample as the sub-corpus benchmarks, and classifies every miss at k=10.

    python scripts/evaluate_full.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safeharbor.config import ROOT  # noqa: E402
from safeharbor.data import load_medquad  # noqa: E402
from safeharbor.evaluation import build_query_set, metrics, pct  # noqa: E402
from safeharbor.tools import SafeHarborTools  # noqa: E402


def classify(q, hits, by_id):
    gold_topics = {by_id[g]["topic"].lower() for g in q["gold"]}
    got_topics = [h["topic"].lower() for h in hits]
    if any(t in gold_topics for t in got_topics):
        if any(h["section"] == q["section"] and h["topic"].lower() in gold_topics for h in hits):
            return "right topic+section, different source doc (near-duplicate NIH page)"
        return "right topic, wrong section"
    return "wrong topic"


def main(n=500):
    tools = SafeHarborTools(airgap=True)
    docs, qa = load_medquad(tools.cfg["paths"]["raw_dir"])
    by_id = {d["doc_id"]: d for d in docs}
    queries = build_query_set(qa, by_id, n=n)
    out, failures = {}, []
    for variant in ("canonical", "paraphrase"):
        ranked, lat = [], []
        for q in queries:
            t0 = time.perf_counter()
            res = tools.search_medical_knowledge(q[variant], k=10)
            lat.append((time.perf_counter() - t0) * 1000)
            ids = [h["doc_id"] for h in res["results"]]
            ranked.append(ids)
            if not set(ids) & q["gold"]:
                failures.append({"variant": variant, "query": q[variant], "section": q["section"],
                                 "gold_topic": by_id[next(iter(q["gold"]))]["topic"],
                                 "top3": [(h["topic"], h["section"]) for h in res["results"][:3]],
                                 "why": classify(q, res["results"], by_id)})
        m = metrics(ranked, [q["gold"] for q in queries])
        m.update(p50_ms=round(pct(lat, 50), 2), p95_ms=round(pct(lat, 95), 2))
        out[variant] = m
        print(variant, m, flush=True)

    out["failure_breakdown"] = {v: Counter(f["why"] for f in failures if f["variant"] == v) for v in ("canonical", "paraphrase")}
    out["failures"] = failures
    res_dir = ROOT / "results"
    (res_dir / "full_index_eval.json").write_text(json.dumps(out, indent=2))
    md = ["# F. Full-index evaluation (production path: e5-small-v2, dense + section router, airgapped)\n",
          f"Index: {tools.index_stats()['collections']} | queries: {n} per variant\n",
          "| variant | recall@1 | recall@5 | recall@10 | mrr@10 | p50 ms (tool call) | p95 ms |",
          "|---|---|---|---|---|---|---|"]
    for v in ("canonical", "paraphrase"):
        m = out[v]
        md.append(f"| {v} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | {m['mrr@10']:.3f} | {m['p50_ms']} | {m['p95_ms']} |")
    md.append("\n## Misses at k=10\n")
    for v, c in out["failure_breakdown"].items():
        md.append(f"**{v}** ({sum(c.values())}): " + "; ".join(f"{k}: {n}" for k, n in c.most_common()))
        md.append("")
    md.append("Examples:\n")
    for f in failures[:12]:
        md.append(f"- *{f['query']}* (gold: {f['gold_topic']} / {f['section']}) → top: {f['top3'][0] if f['top3'] else '-'} — {f['why']}")
    (res_dir / "full_index_eval.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
