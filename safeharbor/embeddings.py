"""Local embedding models.

Each model needs its own instruction prefixes; getting these wrong costs several points of
recall (we measured it - see results/embedding_benchmark.md). All models run on CPU through
sentence-transformers and are loaded from the local HF cache once downloaded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dim: int
    max_seq: int
    query_prefix: str = ""
    doc_prefix: str = ""


MODELS: dict[str, ModelSpec] = {
    "sentence-transformers/all-MiniLM-L6-v2": ModelSpec(
        "sentence-transformers/all-MiniLM-L6-v2", 384, 256),
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": ModelSpec(
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", 384, 512),
    "BAAI/bge-small-en-v1.5": ModelSpec(
        "BAAI/bge-small-en-v1.5", 384, 512,
        query_prefix="Represent this sentence for searching relevant passages: "),
    "intfloat/e5-small-v2": ModelSpec(
        "intfloat/e5-small-v2", 384, 512, query_prefix="query: ", doc_prefix="passage: "),
}


class LocalEmbedder:
    def __init__(self, name: str, device: str = "cpu", batch_size: int = 64, normalize: bool = True):
        from sentence_transformers import SentenceTransformer

        self.spec = MODELS.get(name, ModelSpec(name, 0, 512))
        self.model = SentenceTransformer(name, device=device)
        self.model.max_seq_length = self.spec.max_seq
        self.batch_size = batch_size
        self.normalize = normalize

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def embed_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        texts = [self.spec.doc_prefix + t for t in texts]
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=self.normalize,
                                 show_progress_bar=show_progress, convert_to_numpy=True).astype(np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        texts = [self.spec.query_prefix + t for t in texts]
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=self.normalize,
                                 convert_to_numpy=True).astype(np.float32)
