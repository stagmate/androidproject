"""Agent-facing tool layer: the contract between the Chroma DB and Yankee AI.

The same four tools are exposed over two local transports:
  * MCP (stdio)                         -> server/mcp_server.py
  * OpenAI-style function-calling HTTP  -> server/http_tool_server.py (127.0.0.1 only)

Every response is pseudonymised text + citations (doc_id / chunk_id / section). Raw PHI is
never returned; re-identification is a separate, human-in-the-loop step through PHIVault.
"""
from __future__ import annotations

import time
from functools import cached_property

from .config import load_config
from .embeddings import LocalEmbedder
from .privacy import airgap_enabled, enable_airgap
from .retriever import Retriever, route_section
from .store import get_client

SECTIONS = ["information", "symptoms", "treatment", "inheritance", "frequency", "genetic changes",
            "causes", "exams and tests", "research", "outlook", "susceptibility", "considerations",
            "prevention", "stages", "complications"]

TOOL_SCHEMAS = [
    {
        "name": "search_medical_knowledge",
        "description": ("Search the local NIH-derived medical knowledge base (MedQuAD: diseases, genetics, "
                        "drugs, treatments). Use for any general medical fact. Returns cited passages."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language question or keywords."},
                "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                "section": {"type": "string", "enum": SECTIONS,
                            "description": "Optional hard filter on the knowledge section."},
                "exact_terms": {"type": "boolean", "default": False,
                                "description": ("Set true when the query hinges on exact tokens (gene "
                                                "symbols like CFTR, drug names, codes): adds BM25 to dense.")},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_patient_notes",
        "description": ("Search the de-identified local clinical notes. Names, dates, MRNs, phones and "
                        "addresses are already replaced by placeholders such as [PERSON_1]."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                "condition": {"type": "string", "description": "Optional filter on the primary condition."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_document",
        "description": "Fetch every chunk of one document (knowledge article or patient note) by doc_id.",
        "parameters": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
        },
    },
    {
        "name": "index_stats",
        "description": "Describe the local vector database: collections, sizes, embedding model, privacy mode.",
        "parameters": {"type": "object", "properties": {}},
    },
]


class SafeHarborTools:
    def __init__(self, config_path: str | None = None, airgap: bool | None = None):
        self.cfg = load_config(config_path)
        self.emb = LocalEmbedder(self.cfg["embedding"]["model"], batch_size=self.cfg["embedding"]["batch_size"])
        self.client = get_client(self.cfg["paths"]["chroma_dir"])
        r = self.cfg["retrieval"]
        self.kb = Retriever(self.client.get_collection("medical_kb"), self.emb, r["rrf_k"], r["candidate_pool"],
                            bm25_weight=r.get("bm25_weight", 1.0))
        self.notes = Retriever(self.client.get_collection("patient_notes"), self.emb, r["rrf_k"], r["candidate_pool"])
        self.mode = r["mode"]
        self.exact_mode = r.get("exact_mode", "hybrid")
        self.route = r.get("route", True)
        # Models + index are now in memory; from here on nothing needs the network.
        if airgap if airgap is not None else self.cfg["privacy"]["airgap"]:
            enable_airgap()

    @cached_property
    def names(self):
        return [t["name"] for t in TOOL_SCHEMAS]

    # ------------------------------------------------------------------ tools
    def search_medical_knowledge(self, query: str, k: int = 5, section: str | None = None,
                                 exact_terms: bool = False) -> dict:
        t0 = time.perf_counter()
        where = {"section": section} if section else None
        mode = self.exact_mode if exact_terms else self.mode
        hits = self.kb.search(query, k=k, mode=mode, where=where, route=self.route and section is None)
        return {
            "query": query,
            "routed_section": None if section else route_section(query),
            "retrieval_mode": mode,
            "results": [self._fmt(h) for h in hits],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def search_patient_notes(self, query: str, k: int = 5, condition: str | None = None) -> dict:
        t0 = time.perf_counter()
        where = {"topic": condition} if condition else None
        hits = self.notes.search(query, k=k, mode=self.mode, where=where)
        return {"query": query, "results": [self._fmt(h) for h in hits],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "privacy": "pseudonymised; PHI held in local vault only"}

    def get_document(self, doc_id: str) -> dict:
        col = self.kb.col if doc_id.startswith("mq-") else self.notes.col
        got = col.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        if not got["ids"]:
            return {"doc_id": doc_id, "error": "not found"}
        order = sorted(range(len(got["ids"])), key=lambda i: got["metadatas"][i].get("chunk_index", 0))
        m0 = got["metadatas"][order[0]]
        bodies = [got["documents"][i][int(got["metadatas"][i].get("header_len", 0)):] for i in order]
        return {"doc_id": doc_id, "topic": m0.get("topic"), "section": m0.get("section"),
                "source": m0.get("source"), "chunks": bodies}

    def index_stats(self) -> dict:
        return {
            "collections": {c.name: c.count() for c in self.client.list_collections()},
            "embedding_model": self.cfg["embedding"]["model"],
            "hnsw": self.cfg["hnsw"],
            "retrieval_mode": self.mode,
            "airgap": airgap_enabled(),
        }

    def call(self, name: str, arguments: dict | None = None) -> dict:
        if name not in self.names:
            raise ValueError(f"unknown tool {name!r}")
        return getattr(self, name)(**(arguments or {}))

    @staticmethod
    def _fmt(h) -> dict:
        return {
            "doc_id": h.doc_id,
            "chunk_id": h.chunk_id,
            "topic": h.metadata.get("topic"),
            "section": h.metadata.get("section"),
            "source": h.metadata.get("source"),
            "text": h.body,
            "score": round(h.score, 5),
            "matched_by": sorted(h.sources),
        }
