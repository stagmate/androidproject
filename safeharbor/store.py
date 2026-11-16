"""Chroma DB setup: persistent local client, tuned HNSW collections, batched ingestion."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .privacy import disable_telemetry

disable_telemetry()
import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402


def get_client(path: str | Path | None):
    settings = Settings(anonymized_telemetry=False, allow_reset=True)
    if path is None:
        return chromadb.EphemeralClient(settings=settings)
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=settings)


def create_collection(client, name: str, hnsw: dict, metadata: dict | None = None, reset: bool = True):
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=name,
        # We always pass our own vectors: embeddings are computed by LocalEmbedder so the exact
        # model + prefixes are under our control and identical at ingest and query time.
        embedding_function=None,
        configuration={"hnsw": {
            "space": hnsw.get("space", "cosine"),
            "max_neighbors": int(hnsw.get("max_neighbors", 32)),
            "ef_construction": int(hnsw.get("ef_construction", 200)),
            "ef_search": int(hnsw.get("ef_search", 64)),
        }},
        metadata=metadata or None,
    )


def set_ef_search(collection, ef: int) -> None:
    """Persist a new ef_search in the collection configuration.

    Caveat (measured in scripts/benchmark_hnsw.py): on Chroma 1.x this did not change recall for an
    index already loaded in this process; set ef_search at creation time, or reopen the DB."""
    collection.modify(configuration={"hnsw": {"ef_search": int(ef)}})


def ingest(collection, ids: list[str], texts: list[str], metas: list[dict], embeddings: np.ndarray,
           batch_size: int | None = None) -> float:
    """Add in the largest batches Chroma accepts. Returns seconds spent."""
    client_max = 5000
    try:
        client_max = collection._client.get_max_batch_size()  # type: ignore[attr-defined]
    except Exception:
        pass
    bs = min(batch_size or client_max, client_max)
    t0 = time.perf_counter()
    for i in range(0, len(ids), bs):
        collection.add(
            ids=ids[i:i + bs],
            documents=texts[i:i + bs],
            metadatas=metas[i:i + bs],
            embeddings=embeddings[i:i + bs],
        )
    return time.perf_counter() - t0


def dir_size_mb(path: str | Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def get_all(collection, include: list[str], page: int = 5000) -> dict:
    """``collection.get()`` in pages: a single unpaged get over ~35k rows exceeds SQLite's
    bound-variable limit inside Chroma ("too many SQL variables")."""
    out: dict = {"ids": [], **{k: [] for k in include}}
    offset = 0
    while True:
        got = collection.get(include=include, limit=page, offset=offset)
        if not got["ids"]:
            break
        out["ids"].extend(got["ids"])
        for k in include:
            out[k].extend(list(got[k]))
        offset += len(got["ids"])
    return out
