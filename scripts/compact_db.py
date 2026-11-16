#!/usr/bin/env python
"""Copy the live collections into a fresh Chroma directory (drops orphaned segments and
dead SQLite pages), then swap it in. Vectors are copied as-is, so no re-embedding.

    python scripts/compact_db.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safeharbor.config import load_config  # noqa: E402
from safeharbor.store import create_collection, dir_size_mb, get_all, get_client, ingest  # noqa: E402

cfg = load_config()
src_dir = Path(cfg["paths"]["chroma_dir"])
dst_dir = src_dir.with_name(src_dir.name + "_compact")
shutil.rmtree(dst_dir, ignore_errors=True)
before = dir_size_mb(src_dir)
src, dst = get_client(src_dir), get_client(dst_dir)
for col in src.list_collections():
    got = get_all(col, ["embeddings", "documents", "metadatas"])
    new = create_collection(dst, col.name, cfg["hnsw"], metadata=col.metadata)
    ingest(new, got["ids"], got["documents"], got["metadatas"], got["embeddings"])
    assert new.count() == col.count(), col.name
    print(f"{col.name}: {new.count()} rows copied")
shutil.copy(src_dir / "manifest.json", dst_dir / "manifest.json")
del src, dst
backup = src_dir.with_name(src_dir.name + "_old")
shutil.rmtree(backup, ignore_errors=True)
src_dir.rename(backup)
dst_dir.rename(src_dir)
shutil.rmtree(backup)
m = json.loads((src_dir / "manifest.json").read_text())
m["chroma_dir_mb"] = round(dir_size_mb(src_dir), 1)
(src_dir / "manifest.json").write_text(json.dumps(m, indent=2))
print(f"{before:.1f} MB -> {m['chroma_dir_mb']} MB")
