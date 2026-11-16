"""Retrieval evaluation: query sets, metrics, and helpers shared by the benchmark scripts."""
from __future__ import annotations

import random
import re

import numpy as np

# Paraphrase templates per MedQuAD section: closer to how a clinician or patient actually types
# a question into an agent, and they deliberately avoid the NIH template wording.
_PARA = {
    "symptoms": ["how would I know if someone has {t}", "{t} clinical presentation", "signs to look for in {t}"],
    "treatment": ["how is {t} managed", "therapy options {t}", "what can be done for a patient with {t}"],
    "inheritance": ["can {t} be passed from parent to child", "{t} hereditary pattern"],
    "genetic changes": ["which gene mutations lead to {t}", "molecular basis of {t}"],
    "causes": ["why does {t} happen", "underlying reason for {t}"],
    "exams and tests": ["how do doctors confirm {t}", "diagnostic workup for {t}"],
    "frequency": ["how common is {t}", "prevalence of {t}"],
    "prevention": ["ways to lower the chance of getting {t}", "can {t} be avoided"],
    "susceptibility": ["risk factors for {t}", "who tends to get {t}"],
    "outlook": ["prognosis of {t}", "long-term outlook for people with {t}"],
    "complications": ["what problems can {t} lead to"],
    "stages": ["how is {t} staged"],
    "research": ["ongoing clinical trials for {t}"],
    "information": ["overview of {t}", "explain {t}"],
    "considerations": ["important things to know about {t}"],
}


def build_query_set(qa: list[dict], docs_by_id: dict, n: int = 400, seed: int = 13) -> list[dict]:
    """Sample n questions whose gold docs exist in ``docs_by_id`` and make two variants each:
    the canonical NIH question and a template paraphrase."""
    rng = random.Random(seed)
    pool = [q for q in qa if all(g in docs_by_id for g in q["gold_doc_ids"])]
    rng.shuffle(pool)
    out = []
    for q in pool[:n]:
        topic = docs_by_id[q["gold_doc_ids"][0]]["topic"]
        para = rng.choice(_PARA.get(q["section"], ["tell me about {t}"])).format(t=topic)
        out.append({"canonical": q["question"], "paraphrase": para, "gold": set(q["gold_doc_ids"]),
                    "section": q["section"]})
    return out


def metrics(ranked_doc_ids: list[list[str]], gold: list[set], ks=(1, 5, 10)) -> dict:
    res = {}
    for k in ks:
        res[f"recall@{k}"] = float(np.mean([bool(set(r[:k]) & g) for r, g in zip(ranked_doc_ids, gold)]))
    rr = []
    for r, g in zip(ranked_doc_ids, gold):
        rr.append(next((1.0 / (i + 1) for i, d in enumerate(r[:10]) if d in g), 0.0))
    res["mrr@10"] = float(np.mean(rr))
    return res


def pct(values, p):
    return float(np.percentile(values, p)) if values else float("nan")


def leaked_identifiers(index_texts: list[str], originals: list[str]) -> int:
    """Count raw PHI strings (from the un-redacted source) that still appear anywhere in the index."""
    blob = "\n".join(index_texts)
    return sum(1 for v in originals if v and re.search(re.escape(v), blob))
