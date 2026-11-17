# SafeHarbor-RAG

**Privacy-first clinical retrieval for Yankee AI, built on Chroma DB.**
Our submission to the ISODS 6th World Championship in Data Science & AI (2026/27), track *Healthcare & Life Sciences*.

The repo is a fully local retrieval layer that lets an agent reason over two sources at once: public medical knowledge (NIH / MedQuAD) and private clinical notes. Identifiers are removed **before** anything is embedded. At query time the process runs with outbound networking physically disabled.

**Results on the full 36,466-chunk index** (e5-small-v2, dense + section router, HNSW M=64, all on a CPU, airgapped):

| | recall@1 | recall@10 | MRR@10 | tool-call p50 |
|---|---|---|---|---|
| NIH questions | 0.932 | 0.998 | 0.961 | ≈ 24 ms |
| paraphrased questions | 0.876 | 0.996 | 0.925 | ≈ 23 ms |

The privacy audit found **0 / 2,099** PHI identifiers in the vector DB, outbound networking was blocked, and all **300/300** notes re-identified exactly from the local vault.

- Research writeup: [`WRITEUP.md`](WRITEUP.md)
- Architecture and privacy invariants: [`docs/architecture.md`](docs/architecture.md)
- Benchmark tables: [`results/`](results/)
- Video script: [`docs/VIDEO_SCRIPT.md`](docs/VIDEO_SCRIPT.md)

## Quick start (CPU laptop, ~15 min for the full build)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/ingest.py               # downloads MedQuAD + model once, builds data/chroma
python scripts/audit_privacy.py        # 0 PHI leaks, egress blocked
pytest -q                              # unit + end-to-end tests

# Ask the agent (runs airgapped)
python agent/local_agent.py "Brief me on atrial fibrillation patients and treatment guidance"
```

You can download the prebuilt database instead of building it (see *Dataset* below). Unzip it at the repo root and skip `ingest.py`.

## Connect to Yankee AI

SafeHarbor exposes four tools: `search_medical_knowledge`, `search_patient_notes`, `get_document` and `index_stats`. There are two local transports:

| Transport | Start | Notes |
|---|---|---|
| **MCP (stdio)**, recommended | the agent host launches `python server/mcp_server.py` | No network port at all. Config snippet is in the file header. |
| **HTTP, OpenAI function-calling** | `python server/http_tool_server.py` | Binds to 127.0.0.1 only. `GET /v1/tools` returns the schemas. Call a tool with `POST /v1/tools/{name}`. |

`agent/local_agent.py` is a reference orchestration loop. It runs a deterministic planner by default, or a real tool-calling loop against a local Ollama model (`--llm ollama`). It shows the same tool protocol the Yankee AI agent uses.

## Repository map

```
safeharbor/
  privacy.py      airgap socket guard, telemetry off, PHI redactor
  phi_vault.py    local re-identification map (SQLite, never shipped)
  data.py         MedQuAD loader/normaliser, synthetic PHI notes generator
  chunking.py     sentence-packed, tokenizer-exact chunks with contextual headers
  embeddings.py   local embedding model registry (with correct query/doc prefixes)
  store.py        Chroma persistent client, HNSW config, batched ingestion
  retriever.py    dense / BM25 / hybrid-RRF retrieval, section router, doc collapsing
  tools.py        agent tool contract (schemas + implementations)
  evaluation.py   query sets and metrics
scripts/
  ingest.py            build the database
  benchmark.py         A: embedding models, B: chunking, C: retrieval modes
  benchmark_hnsw.py    D: HNSW M / ef_construction / ef_search sweep vs exact search
  audit_privacy.py     E: PHI leakage, egress, re-identification audit
  evaluate_full.py     F: end-to-end eval on the full index + failure analysis
  compact_db.py        rebuild HNSW graph from stored vectors / reclaim disk
  export_db.sh         package DB + data for Kaggle / Hugging Face
server/   mcp_server.py, http_tool_server.py
agent/    local_agent.py
tests/    privacy, chunking, end-to-end retrieval
```

## Reproducing the results

```bash
python scripts/benchmark.py --docs 6000 --queries 500   # A, B, C -> results/*.md
python scripts/benchmark_hnsw.py --queries 300          # D       -> results/hnsw_benchmark.md
python scripts/audit_privacy.py                         # E       -> results/privacy_audit.json
python scripts/evaluate_full.py                         # F       -> results/full_index_eval.md
python scripts/compact_db.py      # rebuild HNSW with current config.yaml params, no re-embedding
```

All randomness is seeded. Each benchmark runs on a CPU-only machine.

## Dataset

`bash scripts/export_db.sh` writes `dist/safeharbor-chroma-db.zip`. The archive contains:

- the persistent Chroma DB with `manifest.json`
- the MedQuAD snapshot
- the evaluation questions
- the synthetic notes, both before and after redaction
- all result tables

The PHI vault is deliberately excluded.

## Data and licences

- MedQuAD: Ben Abacha & Demner-Fushman, *A question-entailment approach to question answering*, BMC Bioinformatics 2019. CC BY 4.0. The answers are NIH content.
- The clinical notes are **synthetic**, generated by `safeharbor/data.py`. No real patient data is used anywhere.
- Code: MIT.
