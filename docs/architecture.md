# Architecture

```
                         ┌──────────────────────── LOCAL MACHINE (airgapped at query time) ────────────────────────┐
                         │                                                                                          │
  MedQuAD (NIH, public)──┼─► normalise + dedupe ─┐                                                                  │
                         │                       ├─► sentence-packed chunking ─► bge-small-en-v1.5 ─► Chroma DB     │
  EHR notes (PHI) ───────┼─► PHI redaction ──────┘    (tokenizer-exact,         (CPU, 384-d,        (persistent,     │
                         │        │                    contextual header)         L2-normalised)      HNSW cosine,    │
                         │        ▼                                                                   M=32, efC=200)  │
                         │   PHI vault (SQLite) ◄──── re-identify for authorised viewer only          │               │
                         │                                                                           ▼               │
                         │                                         Retriever: dense ⊕ BM25 ─► RRF ─► section prior   │
                         │                                                                           │               │
                         │        Yankee AI agent ◄── MCP (stdio) / HTTP 127.0.0.1 ◄── SafeHarborTools ◄──────┘        │
                         │        (local LLM)            search_medical_knowledge · search_patient_notes · get_document │
                         └──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Privacy invariants** (each one is tested or audited):

| # | Invariant | Enforced by | Verified by |
|---|---|---|---|
| 1 | Raw PHI is never embedded or stored in Chroma | `PHIRedactor` runs before chunking | `scripts/audit_privacy.py` (0 leaks) |
| 2 | No outbound network at query time | `privacy.enable_airgap()` patches `socket` | `tests/test_privacy.py`, audit |
| 3 | No telemetry | `ANONYMIZED_TELEMETRY=False`, HF offline | audit |
| 4 | Tool server is unreachable from the LAN | MCP over stdio; HTTP binds to loopback only | `server/http_tool_server.py` refuses other hosts |
| 5 | The LLM sees pseudonyms only | tools return redacted text; vault isn't exposed as a tool | code review: `safeharbor/tools.py` |
| 6 | The shared dataset can't be re-identified | `export_db.sh` excludes `phi_vault.sqlite` | archive listing |
