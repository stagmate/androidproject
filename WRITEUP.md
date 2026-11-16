# SafeHarbor-RAG
### Privacy-first clinical retrieval for Yankee AI: a de-identified, airgapped Chroma DB that lets a local agent reason over patient notes and NIH medical knowledge together

**Track:** Healthcare & Life Sciences
**Team:** _<name 1 — affiliation/status>_, _<name 2>_, _<name 3>_
**Links:** Code: _<GitHub URL>_ · Vector DB + data: _<Kaggle / Hugging Face URL>_ · Video: _<YouTube URL>_

---

## 1. Problem: clinicians need agents, and agents leak

A physician preparing for a clinic session wants to ask: *"Which of my atrial-fibrillation patients are poorly controlled, and what does current guidance say about treatment and complications?"* Answering this means joining two sources:

1. **Private records** (progress notes, discharge summaries). These are full of the identifiers HIPAA calls PHI: names, birth dates, record numbers, phones and addresses.
2. **Public medical knowledge** (NIH consumer and genetics references).

Cloud LLM agents can do the join, but every tool call ships PHI to a third party. Local RAG is the obvious fix, but "local" by itself is not a guarantee:

- **Embeddings are not anonymous.** Inversion attacks recover much of the source text from sentence embeddings (Morris et al., *Vec2Text*, 2023). A vector DB built from raw notes is itself PHI.
- **Libraries phone home.** Vector DBs and model hubs ship with telemetry and auto-download on by default.
- **Agents are unpredictable.** A tool-calling LLM can be prompt-injected into fetching a URL.

**SafeHarbor-RAG** is our answer. It is a Chroma DB retrieval layer for Yankee AI built around four invariants that we enforce in code and verify with tests and an audit script:

| Invariant | Mechanism |
|---|---|
| PHI is never embedded or stored in the vector DB | Deterministic de-identification **before** chunking; reversible only through a separate local vault |
| Nothing leaves the machine at query time | An airgap guard patches `socket` so any non-loopback connection or DNS lookup raises |
| The agent sees pseudonyms only | Tools return `[PERSON_1]`-style text. Re-identification is a human-side step, never an LLM tool |
| The tool server is unreachable from the network | MCP over stdio (no port at all), or HTTP bound to 127.0.0.1 only |

## 2. Data

| Collection | Source | Docs | Chunks | Purpose |
|---|---|---|---|---|
| `medical_kb` | **MedQuAD** (Ben Abacha & Demner-Fushman 2019, CC BY 4.0): 16,407 Q/A pairs from 12 NIH sites (MedlinePlus, GARD, GHR, NIDDK, NCI, …) | 15,798 after exact-duplicate merge | 35,811 | Public medical knowledge |
| `patient_notes` | 300 **synthetic** clinical notes (`safeharbor/data.py`, seeded), each with 8 classes of PHI | 300 | _<from manifest>_ | Stand-in for an EHR export |

Each MedQuAD answer is normalised into a document with `topic` (the disease, gene or drug, extracted from the question by 19 template regexes) and `section` (NIH question type: symptoms, treatment, inheritance, genetic changes, …, 15 types). When several NIH sites answer the same question, all of their answers count as relevant in evaluation.

**Why synthetic notes?** Real notes cannot be published, and the competition requires a public dataset. The generator produces realistic structure (demographics block, history, medications, findings, plan, emergency contact), so the redaction path is exercised end to end, and the ground-truth PHI is known exactly, which is what makes a leakage audit possible.

## 3. Vector database engineering

### 3.1 De-identification before embedding
`PHIRedactor` (no external NLP model, fully deterministic) removes the HIPAA Safe-Harbor identifier classes that occur in free text:

- names, found via cue patterns (`Patient:`, `Mr./Dr.`, relation fields) plus names known from the structured EHR header
- dates, phone numbers, email, SSN, MRN, street address and ZIP

Placeholders are **stable within a document**: every mention of Ms. Lopez becomes `[PERSON_1]`, and a bare "Lopez" becomes `[PERSON_1.2]`. The agent can still reason ("[PERSON_1]'s INR…"), and the mapping goes to `phi_vault.sqlite`. That vault is never exposed as a tool and is excluded from the published dataset, so the shipped index is irreversibly de-identified.

### 3.2 Chunking: sentence-packed, tokenizer-exact, with contextual headers
- Chunks are packed from **whole sentences**. NIH bullet runs are split on the bullet, so a symptom list can't form one 400-token "sentence".
- The size budget is counted in **the embedding model's own tokenizer**. Character-based splitters silently overflow small models: all-MiniLM-L6-v2 truncates at 256 word-pieces, and we found MedQuAD answers of 600+ tokens.
- Every chunk is prefixed with a **contextual header**: `"Cystic Fibrosis | inheritance: …"`. Without it, a chunk like *"It is inherited in an autosomal recessive pattern"* is unanchored.

**Ablation** (e5-small-v2, dense only, 6,000-doc corpus, 500 queries; see `results/chunking_benchmark.md`):

| Strategy | Chunks | R@1 canonical | R@1 paraphrase | MRR@10 para. |
|---|---|---|---|---|
| Naive 1000-char / 200 overlap, no header (framework default) | 11,532 | 0.758 | 0.714 | 0.781 |
| 192 tok, overlap 1, **no header** | 12,831 | 0.744 | 0.704 | 0.772 |
| 96 tok, overlap 1, header | 31,510 | 0.888 | 0.878 | 0.926 |
| **192 tok, overlap 1, header (ours)** | 13,556 | 0.882 | 0.868 | 0.921 |
| 192 tok, overlap 0, header | 12,670 | 0.888 | 0.886 | 0.933 |
| 384 tok, overlap 1, header | 7,998 | 0.866 | 0.882 | 0.930 |

**Finding:** the contextual header is the single largest lever in the whole pipeline: **+14 to +16 points of R@1**. Chunk size between 96 and 384 tokens matters far less. 96 tokens needs 2.3× the vectors for no gain. Sentence overlap is within noise (±1.5 pt at n = 500). We keep one sentence of overlap because it protects multi-sentence instructions (dose + contraindication), which this benchmark doesn't measure.

### 3.3 Embedding model selection
Four CPU-friendly 384-d models, same chunks (`results/embedding_benchmark.md`):

| Model | R@1 canon. | R@1 para. | MRR para. | Embed throughput (chunks/s, 4 CPU cores) |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.776 | 0.742 | 0.849 | 92 |
| multi-qa-MiniLM-L6-cos-v1 | 0.802 | 0.750 | 0.852 | 96 |
| bge-small-en-v1.5 | 0.864 | 0.840 | 0.906 | 50 |
| **e5-small-v2** | **0.884** | **0.870** | **0.923** | 48 |

We chose **e5-small-v2**. It is about 2× slower to embed than MiniLM, but ingestion is a one-off cost; query embedding is a single short string (~5 ms), and the recall gain is +11 to +13 points. Model-specific prefixes (`query:` / `passage:` for e5; the instruction prefix for bge) live in a registry, so ingest and query can never disagree.

### 3.4 Retrieval: dense, BM25, hybrid, and a section router
MedQuAD's hard case isn't finding the right *disease*. It's finding the right *section* of that disease, because ~10 near-identical chunks per topic all mention it. We added a **section router**: 13 regex intents (e.g. "how do doctors confirm" → `exams and tests`) apply a *soft* score bonus to matching sections. It is never a hard filter, so a mis-route cannot remove the correct answer. Results are in `results/retrieval_modes.md`:

| Mode | R@1 canon. | R@1 para. | MRR para. | p50 latency |
|---|---|---|---|---|
| BM25 | 0.558 | 0.510 | 0.694 | 11.9 ms |
| Dense | 0.882 | 0.870 | 0.923 | 6.0 ms |
| Hybrid (RRF, equal weights) | 0.800 | 0.798 | 0.879 | 20.3 ms |
| Hybrid + router | 0.896 | 0.898 | 0.938 | 19.8 ms |
| Hybrid (BM25 weight 0.3) + router | 0.916 | **0.936** | **0.960** | 18.9 ms |
| **Dense + router (default)** | **0.930** | 0.930 | 0.959 | **6.7 ms** |

**Findings.**
1. **A negative result:** textbook equal-weight RRF *hurts* by 8 points. BM25 is a much weaker ranker on this corpus, and RRF gives it an equal vote. Down-weighting BM25 to 0.3 recovers the loss.
2. The **router adds +4.8 to +6 points** of R@1 at negligible cost.
3. **Dense + router is the best default:** the top average R@1 (0.930) at a third of hybrid's latency. Weighted hybrid is exposed to the agent as an opt-in `exact_terms=true` flag, for queries that hinge on gene symbols or drug codes, where lexical match matters.

### 3.5 HNSW tuning in Chroma
_<fill from `results/hnsw_benchmark.md`: M ∈ {8,16,32,48} × ef_search ∈ {10…256}, recall@10 vs exact brute-force, p50/p95 latency, build time, disk>_

Chosen operating point: `max_neighbors (M)=32, ef_construction=200, ef_search=64`, set through Chroma's collection `configuration`. `ef_search` can be changed at runtime with `collection.modify()` without rebuilding the index.

### 3.6 Full index
_<from `data/chroma/manifest.json`: chunks, embed/index seconds, on-disk MB>_

## 4. Yankee AI integration

```
Yankee AI agent (local LLM)
   │  MCP over stdio  (no listening port)          or   HTTP, OpenAI tool schema, 127.0.0.1:8765
   ▼
SafeHarborTools ── search_medical_knowledge(query, k, section?, exact_terms?)
                ── search_patient_notes(query, k, condition?)
                ── get_document(doc_id)
                ── index_stats()
   ▼
Retriever (dense + router | weighted hybrid) ──► Chroma PersistentClient (HNSW cosine)
                       process runs under airgap(): any non-loopback socket raises
```

Integration is through standard agent-tool protocols, so nothing in Yankee AI has to change:

- **MCP:** register `python server/mcp_server.py` as a stdio server in Yankee AI's tool configuration (snippet in the file header).
- **HTTP:** `GET /v1/tools` returns OpenAI function-calling schemas, and `POST /v1/tools/{name}` executes a tool.

Every result carries `doc_id`, `chunk_id`, `section`, `source` and `matched_by`, so the agent can cite. Notes come back pseudonymised.

**Demo workflow (video, Part 2).** The question *"Brief me on our atrial-fibrillation patients and current treatment guidance"* runs this chain:

1. `search_patient_notes` → 3 de-identified notes
2. condition extraction
3. for each condition, `search_medical_knowledge(section=treatment)` and `search_medical_knowledge(section=complications)`
4. a cited brief

Optionally, the authorised viewer re-identifies the result locally from the vault. `agent/local_agent.py` reproduces the same chain without Yankee AI: deterministically, or with a local Ollama model in a real tool-calling loop.

## 5. Privacy evaluation
_<fill from `results/privacy_audit.json`>_

- PHI leakage: every ground-truth identifier from the 300 raw notes was searched for in every stored document and metadata field. **Leaked: _N_ of _M_.**
- Egress: an outbound connection attempted inside `airgap()` was blocked.
- Re-identification round-trip: _x_/300 notes were restored byte-exact from the vault.
- Chroma telemetry: off.

## 6. Limitations and honest notes
- **Evaluation queries.** "Canonical" queries are MedQuAD's own NIH questions. "Paraphrase" queries are template rewrites that avoid the NIH wording but contain the topic name. Neither set is real clinician language. A clinician-written test set is the most important next step.
- **Recall@10 ceiling (~0.96 canonical).** _<fill after failure analysis>_
- **Redaction is rule-based.** It reaches 0 leaks on our synthetic notes, whose PHI formats we know. Real notes have more variety: nicknames, free-text dates, rare names without cues. In production we would add a local NER model (e.g. a de-identification transformer) and keep the audit as a regression gate.
- **Synthetic notes** test the privacy pipeline, not clinical retrieval quality.

## 7. Impact beyond healthcare
The pattern is: de-identify before embedding, keep the vault separate, airgap the query path, and expose only pseudonymised tools to the agent. Nothing in it is specific to medicine. Swapping the redactor's identifier classes adapts it to privileged legal files (party names, case numbers), financial records (account numbers) or student data (FERPA).

## 8. Reproducibility
Everything runs on a 4-core CPU laptop with no GPU and no API keys:
```
python scripts/ingest.py            # ≈ 15 min full build
python scripts/benchmark.py         # experiments A–C
python scripts/benchmark_hnsw.py    # experiment D
python scripts/audit_privacy.py     # experiment E
pytest -q                           # 12 tests
```
All randomness is seeded, and model and dataset versions are pinned in `data/chroma/manifest.json`.

## References
- Ben Abacha, A., & Demner-Fushman, D. (2019). A question-entailment approach to question answering. *BMC Bioinformatics* 20, 511.
- Morris, J. X., et al. (2023). Text embeddings reveal (almost) as much as text. *EMNLP*.
- Cormack, G. V., Clarke, C. L., & Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR*.
- Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using HNSW graphs. *IEEE TPAMI*.
- Wang, L., et al. (2022). Text embeddings by weakly-supervised contrastive pre-training (E5). *arXiv:2212.03533*.
- U.S. HHS. Guidance regarding methods for de-identification of PHI (Safe Harbor method), 45 CFR §164.514(b).
