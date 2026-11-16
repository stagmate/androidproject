"""Query-time retrieval: dense (Chroma HNSW), sparse (BM25) and hybrid (Reciprocal Rank Fusion),
with an optional section router and chunk -> document collapsing.

Why hybrid: medical questions are full of rare exact tokens (drug names, gene symbols such as
"CFTR", syndrome eponyms) that small embedding models blur together. BM25 nails those; the
dense model handles paraphrase ("can't catch my breath" ~ "dyspnea"). RRF fuses the two ranks
without any score calibration.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

_STOP = set("""a an and are as at be by can do does for from how i if in is it its of on or
should the their there these this to was what when where which who why will with you your
my me am have has had not""".split())
_TOK = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def bm25_tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


# Section router: maps question cues onto MedQuAD section labels (qtype).
_ROUTES = [
    ("symptoms", r"\b(symptom|signs?\b|feel like|presentation)"),
    ("treatment", r"\b(treat|therap|cure|medication|drug|manag)"),
    ("inheritance", r"\b(inherit|hereditary|passed? (on|down|from parent))"),
    ("genetic changes", r"\b(genes?\b|genetic|mutation|variant)"),
    ("causes", r"\b(causes?\b|caused|why does|trigger)"),
    ("exams and tests", r"\b(diagnos|tests?\b|exam|screen|workup)"),
    ("frequency", r"\b(how (many|common)|prevalen|incidence)"),
    ("prevention", r"\b(prevent|avoid|lower the chance|reduce (the )?risk)"),
    ("susceptibility", r"\b(at risk|risk factor|who (gets|is likely|tends))"),
    ("outlook", r"\b(outlook|prognosis|life expectancy|survival)"),
    ("complications", r"\b(complication|problems can .* lead)"),
    ("stages", r"\b(stages?\b|staged)"),
    ("research", r"\b(research|clinical trials?)"),
]
_ROUTES_RX = [(sec, re.compile(rx, re.I)) for sec, rx in _ROUTES]


def route_section(query: str) -> str | None:
    for sec, rx in _ROUTES_RX:
        if rx.search(query):
            return sec
    return None


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict
    score: float
    sources: dict = field(default_factory=dict)  # {"dense": rank, "bm25": rank}

    @property
    def body(self) -> str:
        return self.text[int(self.metadata.get("header_len", 0)):]


class Retriever:
    def __init__(self, collection, embedder, rrf_k: int = 60, candidate_pool: int = 50,
                 build_bm25: bool = True):
        self.col = collection
        self.emb = embedder
        self.rrf_k = rrf_k
        self.pool = candidate_pool
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._metas: list[dict] = []
        self.bm25 = None
        if build_bm25:
            self._build_bm25()

    # -- sparse ----------------------------------------------------------------
    def _build_bm25(self):
        from rank_bm25 import BM25Okapi

        got = self.col.get(include=["documents", "metadatas"])
        self._ids, self._docs, self._metas = got["ids"], got["documents"], got["metadatas"]
        self._pos = {cid: i for i, cid in enumerate(self._ids)}
        self._meta_cols: dict[str, np.ndarray] = {}
        self.bm25 = BM25Okapi([bm25_tokenize(d) for d in self._docs]) if self._docs else None

    def _bm25_search(self, query: str, n: int, where: dict | None) -> list[tuple[str, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(bm25_tokenize(query))
        if where:
            mask = np.ones(len(scores), dtype=bool)
            for k, v in where.items():
                if k not in self._meta_cols:
                    self._meta_cols[k] = np.array([m.get(k) for m in self._metas], dtype=object)
                mask &= self._meta_cols[k] == v
            scores = np.where(mask, scores, -np.inf)
        top = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]

    # -- dense -----------------------------------------------------------------
    def _dense_search(self, qvec: np.ndarray, n: int, where: dict | None):
        res = self.col.query(query_embeddings=[qvec.tolist()], n_results=n, where=where or None,
                             include=["documents", "metadatas", "distances"])
        return list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]))

    # -- public ----------------------------------------------------------------
    def search(self, query: str, k: int = 5, mode: str = "hybrid", where: dict | None = None,
               route: bool = False, collapse_docs: bool = True, qvec: np.ndarray | None = None) -> list[Hit]:
        """Return the top-k hits. ``route=True`` boosts chunks whose section matches the
        question intent (a soft prior, never a hard filter, so a mis-route cannot zero recall)."""
        pool = max(self.pool, k * 4)
        by_id: dict[str, Hit] = {}
        fused: dict[str, float] = {}

        if mode in ("dense", "hybrid"):
            if qvec is None:
                qvec = self.emb.embed_queries([query])[0]
            for rank, (cid, doc, meta, dist) in enumerate(self._dense_search(qvec, pool, where)):
                by_id[cid] = Hit(cid, meta.get("doc_id", cid.split("::")[0]), doc, meta, 0.0, {"dense": rank + 1})
                fused[cid] = fused.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1) if mode == "hybrid" else 1 - dist)

        if mode in ("bm25", "hybrid"):
            for rank, (cid, s) in enumerate(self._bm25_search(query, pool, where)):
                if cid not in by_id:
                    i = self._pos[cid]
                    by_id[cid] = Hit(cid, self._metas[i].get("doc_id", cid.split("::")[0]), self._docs[i],
                                     self._metas[i], 0.0, {})
                by_id[cid].sources["bm25"] = rank + 1
                fused[cid] = fused.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1) if mode == "hybrid" else s)

        if route:
            sec = route_section(query)
            if sec:
                bonus = 1.0 / self.rrf_k if mode == "hybrid" else 0.05
                for cid, h in by_id.items():
                    if h.metadata.get("section") == sec:
                        fused[cid] += bonus * 0.5

        ranked = sorted(fused, key=fused.get, reverse=True)
        out: list[Hit] = []
        seen_docs: set[str] = set()
        for cid in ranked:
            h = by_id[cid]
            h.score = fused[cid]
            if collapse_docs:
                if h.doc_id in seen_docs:
                    continue
                seen_docs.add(h.doc_id)
            out.append(h)
            if len(out) >= k:
                break
        return out


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return r, (time.perf_counter() - t0) * 1000
