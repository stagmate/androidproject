#!/usr/bin/env python
"""Build the SafeHarbor Chroma DB from scratch.

    python scripts/ingest.py                 # full MedQuAD + 300 synthetic notes
    python scripts/ingest.py --limit 2000    # quick build for a laptop demo

Pipeline per collection:
  load -> (notes only) PHI redaction + vault -> sentence-packed chunking with headers
  -> local embedding (CPU) -> Chroma HNSW collection (tuned M / ef) -> manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safeharbor.chunking import iter_chunks, make_token_counter  # noqa: E402
from safeharbor.config import load_config  # noqa: E402
from safeharbor.data import generate_notes, load_medquad  # noqa: E402
from safeharbor.embeddings import LocalEmbedder  # noqa: E402
from safeharbor.phi_vault import PHIVault  # noqa: E402
from safeharbor.privacy import PHIRedactor  # noqa: E402
from safeharbor.store import create_collection, dir_size_mb, get_client, ingest  # noqa: E402


def build(cfg: dict, limit: int | None, n_notes: int) -> dict:
    paths, ecfg, ccfg = cfg["paths"], cfg["embedding"], cfg["chunking"]
    emb = LocalEmbedder(ecfg["model"], batch_size=ecfg["batch_size"], normalize=ecfg["normalize"])
    count = make_token_counter(emb.tokenizer)
    client = get_client(paths["chroma_dir"])
    manifest = {"embedding_model": ecfg["model"], "chunking": ccfg, "hnsw": cfg["hnsw"], "collections": {}}

    # ---------------- public knowledge base ----------------
    docs, qa = load_medquad(paths["raw_dir"])
    if limit:
        docs = docs[:limit]
    Path(paths["raw_dir"], "eval_questions.jsonl").write_text("\n".join(json.dumps(q) for q in qa) + "\n")

    # ---------------- private notes (PHI) ----------------
    notes = generate_notes(n_notes)
    Path(paths["raw_dir"], "synthetic_notes_with_phi.jsonl").write_text(
        "\n".join(json.dumps({k: v for k, v in n.items() if not k.startswith("_")}) for n in notes) + "\n")
    vault = PHIVault(paths["phi_vault"])
    red = PHIRedactor()
    redacted = []
    n_phi = 0
    for n in notes:
        r = red.redact(n["text"], known_names=n["_known_names"])
        vault.store(n["doc_id"], r.mapping)
        n_phi += r.n_redacted
        redacted.append({**{k: v for k, v in n.items() if not k.startswith("_")}, "text": r.text, "phi_redacted": r.n_redacted})
    Path(paths["raw_dir"], "notes_redacted.jsonl").write_text("\n".join(json.dumps(n) for n in redacted) + "\n")

    for name, items in (("medical_kb", docs), ("patient_notes", redacted)):
        t0 = time.perf_counter()
        ids, texts, metas = [], [], []
        for cid, text, meta in iter_chunks(items, count, ccfg["max_tokens"], ccfg["overlap_sentences"], ccfg["header"]):
            ids.append(cid); texts.append(text); metas.append(meta)
        t_chunk = time.perf_counter() - t0
        print(f"[{name}] {len(items)} docs -> {len(ids)} chunks ({t_chunk:.1f}s)")
        t0 = time.perf_counter()
        vecs = emb.embed_documents(texts, show_progress=True)
        t_emb = time.perf_counter() - t0
        col = create_collection(client, name, cfg["hnsw"],
                                metadata={"embedding_model": ecfg["model"], "privacy": "local-only"})
        t_add = ingest(col, ids, texts, metas, vecs)
        manifest["collections"][name] = {
            "documents": len(items), "chunks": len(ids),
            "seconds": {"chunk": round(t_chunk, 1), "embed": round(t_emb, 1), "index": round(t_add, 1)},
            "chunks_per_sec_embed": round(len(ids) / max(t_emb, 1e-9), 1),
        }
        print(f"[{name}] embedded in {t_emb:.1f}s, indexed in {t_add:.1f}s")

    manifest["phi"] = {"notes": len(notes), "identifiers_redacted": n_phi, "vault_rows": vault.count()}
    manifest["chroma_dir_mb"] = round(dir_size_mb(paths["chroma_dir"]), 1)
    Path(paths["chroma_dir"], "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--limit", type=int, help="only index the first N MedQuAD documents")
    ap.add_argument("--notes", type=int, default=300)
    a = ap.parse_args()
    build(load_config(a.config), a.limit, a.notes)
