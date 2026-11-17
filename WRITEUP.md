# SafeHarbor-RAG
### Privacy-first clinical retrieval for Yankee AI: a de-identified, airgapped Chroma DB that lets a local agent reason over patient notes and NIH medical knowledge together

**Track:** Healthcare & Life Sciences
**Team:** _<name 1 — affiliation/status>_, _<name 2>_, _<name 3>_
**Links:** Code: _<GitHub URL>_ · Vector DB + data: _<Kaggle / Hugging Face URL>_ · Video: _<YouTube URL>_

---

## Abstract
SafeHarbor-RAG is a fully local retrieval layer for the Yankee AI agent platform, built on Chroma DB. It joins two sources an agent needs: private clinical notes and public NIH medical knowledge (MedQuAD). The privacy guarantee comes from four mechanisms:

- identifiers are removed **before** embedding (embeddings can be inverted);
- the re-identification map is kept in a separate local vault that the LLM can never reach;
- at query time the process runs with outbound networking physically disabled;
- tools are exposed over MCP stdio (no port) or loopback-only HTTP.

On the full 36,466-chunk index, the production path reaches **recall@1 = 0.932 and recall@10 = 0.998** on NIH questions at about 24 ms per tool call on a 4-core CPU. The privacy audit finds **0 of 2,099 identifiers** in the database. Getting there produced five engineering findings that we document in full, including two failures:

1. a contextual header on each chunk is worth +15 points of recall@1;
2. textbook equal-weight hybrid RRF *hurts* on this corpus;
3. a cheap section router is the best ranking signal we found;
4. a common deduplication shortcut silently erased 548 disease-specific documents;
5. the standard synthetic ANN benchmark overstates HNSW recall on real queries, and tuning on real queries changed our index parameters.

## 1. Problem: clinicians need agents, and agents leak

A physician preparing a clinic session wants to ask: *"Which of my atrial-fibrillation patients are poorly controlled, and what does guidance say about treatment?"* Answering this means joining **private records** with **public medical knowledge**. Progress notes and discharge summaries are full of HIPAA identifiers: names, birth dates, MRNs, phone numbers, addresses.

Cloud LLM agents can make that join, but every tool call ships PHI to a third party. Running RAG locally is the obvious fix, but "local" is not by itself a guarantee:

- **Embeddings are not anonymous.** Inversion attacks recover much of the source text from sentence embeddings (Morris et al., 2023). A vector DB built from raw notes *is* PHI.
- **Libraries phone home.** Vector-DB telemetry and model-hub auto-downloads are on by default.
- **Agents are unpredictable.** A prompt-injected agent may try to fetch a URL with data in it.

SafeHarbor-RAG enforces four invariants in code and verifies each one with tests or an audit script:

| Invariant | Mechanism | Verified by |
|---|---|---|
| Raw PHI is never embedded or stored in Chroma | Deterministic de-identification **before** chunking | `audit_privacy.py`: 0 / 2,099 leaked |
| Nothing leaves the machine at query time | `airgap()` patches `socket`; any non-loopback connect or DNS lookup raises | unit test + audit |
| The agent sees pseudonyms only | Tools return `[PERSON_1]`-style text; the vault is not a tool | code (`safeharbor/tools.py`) |
| The tool server is unreachable from the network | MCP over stdio (no port), or HTTP that refuses non-loopback binds | `server/http_tool_server.py` |

## 2. Data

| Collection | Source | Docs | Chunks |
|---|---|---|---|
| `medical_kb` | **MedQuAD** (Ben Abacha & Demner-Fushman 2019, CC BY 4.0): 16,407 Q/A pairs from 12 NIH sites (MedlinePlus, GARD, Genetics Home Reference, NIDDK, NCI, …) | 16,346 | 36,466 |
| `patient_notes` | 300 **synthetic** clinical notes (seeded generator) containing 8 PHI types; 3,260 identifiers in total | 300 | 300 |

**Normalisation.** Each answer becomes a document with three fields:

- `topic`: the disease, gene or drug, taken from the question by 19 template regexes
- `section`: one of NIH's 15 question types (symptoms, treatment, inheritance, genetic changes, …)
- `source`

**Deduplication lesson (finding 4).** We first merged documents with identical answer text. That looked harmless, but NIH reuses some paragraphs *verbatim across diseases*: the generic autosomal-recessive inheritance text appears under hundreds of conditions. Merging those kept only the *first* disease's label, which erased 548 disease-specific documents. Every other condition's "Is X inherited?" question became unanswerable, which produced a hard recall@10 ceiling of about 0.95–0.96 that every embedding model hit identically.

The failure analysis exposed it: all 25 canonical misses at k = 10 pointed at a chunk labelled "Nijmegen breakage syndrome" (`results/full_index_eval_v1_answer_dedup.md`). Deduplicating on **(topic, answer)** fixed it. Full-index R@10 went from 0.950 to 0.998 and R@1 from 0.884 to 0.932. All tables below use the corrected data.

**Why synthetic notes?** Real notes cannot be published, and the competition requires a public dataset. The generator produces realistic structure (demographics block, history, medications, findings, plan, emergency contact), so the redaction path is exercised end to end. Because the ground-truth PHI is known exactly, the leakage audit is exact too.

## 3. Vector database engineering

### 3.1 De-identification before embedding
`PHIRedactor` is deterministic and dependency-free. It targets the HIPAA Safe-Harbor identifier classes that occur in free text:

- names, found via cue patterns (`Patient:`, `Mr./Dr.`, relation fields) plus names from the structured EHR header
- dates, phone numbers, email, SSN, MRN, street addresses and ZIP codes

Placeholders are **stable within a note**. Every mention of a patient becomes `[PERSON_1]`, and a bare given name becomes `[PERSON_1.1]`, so the agent can still reason about "the patient" across sentences. The placeholder→value map goes to `phi_vault.sqlite`. That file is never exposed as a tool and is excluded from the published dataset, so the shipped index is irreversibly de-identified.

The leakage audit searches for every one of the 2,099 unique ground-truth identifiers in every stored document *and* metadata field. It finds **0 leaks**, and a second pass for bare given names and surnames is also clean. The vault restores all **300/300** notes byte-exact for authorised local display.

### 3.2 Chunking: sentence-packed, tokenizer-exact, with contextual headers
- Chunks are packed from **whole sentences**. NIH bullet runs are split on the bullet, so a long symptom list cannot form one giant "sentence".
- The size budget is counted in **the embedding model's own tokenizer**. Character-based splitters silently overflow small models: all-MiniLM-L6-v2 truncates at 256 word-pieces, and MedQuAD answers reach 600+.
- Every chunk is prefixed with a **contextual header**: `"Cystic Fibrosis | inheritance: …"`.

**Ablation** (e5-small-v2, dense only, 6,000-document sub-corpus, 500 queries × 2 variants; `results/chunking_benchmark.md`):

| Strategy | Chunks | R@1 canonical | R@1 paraphrase | R@10 para. |
|---|---|---|---|---|
| Naive 1000-char / 200 overlap, no header (common framework default) | 11,614 | 0.766 | 0.730 | 0.896 |
| 192 tok, overlap 1, **no header** | 12,900 | 0.766 | 0.710 | 0.880 |
| 96 tok, overlap 1, header | 31,537 | 0.926 | 0.878 | 0.996 |
| **192 tok, overlap 1, header (ours)** | 13,654 | 0.918 | 0.870 | 0.996 |
| 192 tok, overlap 0, header | 12,721 | 0.922 | 0.888 | 0.998 |
| 384 tok, overlap 1, header | 8,080 | 0.908 | 0.880 | 0.998 |

**Finding 1: the contextual header is the largest single lever in the pipeline, worth +15 to +16 points of R@1.** A chunk reading *"This condition is inherited in an autosomal recessive pattern"* is identical across hundreds of diseases. Only the header disambiguates it.

Chunk size between 96 and 384 tokens matters much less: all configurations sit within about ±2 points, which is close to noise at n = 500 (SE ≈ 1.2 pt). 96 tokens needs 2.3× as many vectors for no gain. Sentence overlap is also within noise. We keep one sentence of overlap because it protects multi-sentence instructions (a dose followed by its contraindication), which this benchmark doesn't measure.

### 3.3 Embedding model selection
Four CPU-friendly 384-dimensional models, same chunks (`results/embedding_benchmark.md`):

| Model | R@1 canon. | R@1 para. | MRR para. | Embed throughput¹ |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.810 | 0.746 | 0.856 | 92 chunks/s |
| multi-qa-MiniLM-L6-cos-v1 | 0.842 | 0.762 | 0.857 | 96 chunks/s |
| bge-small-en-v1.5 | 0.894 | 0.848 | 0.911 | 50 chunks/s |
| **e5-small-v2** | **0.920** | **0.870** | **0.926** | 48 chunks/s |

¹ 4 CPU cores, measured on an idle machine (first run).

We chose **e5-small-v2**. It embeds about 2× slower than MiniLM, but ingestion is a one-off cost (about 14 minutes for the full corpus), a query is one short string, and the recall gain is +11 to +12 points. Model-specific prefixes (`query:` / `passage:` for e5; the instruction prefix for bge) live in one registry, so ingest and query can never disagree.

### 3.4 Retrieval: dense, BM25, hybrid, and a section router
After the header fix, the hard part of MedQuAD isn't finding the right *disease*. It's choosing the right *section* among ~10 near-identical chunks about that disease. We added a **section router**: 13 regex intents (e.g. "how do doctors confirm" → `exams and tests`) apply a *soft* score bonus to matching sections. It is never a hard filter, so a mis-route cannot remove the correct answer. Results are in `results/retrieval_modes.md`:

| Mode | R@1 canon. | R@1 para. | MRR canon. | p50 |
|---|---|---|---|---|
| BM25 | 0.620 | 0.542 | 0.764 | 10.9 ms |
| Dense | 0.916 | 0.868 | 0.953 | 6.9 ms |
| Hybrid (RRF, equal weights) | 0.836 | 0.792 | 0.905 | 20.0 ms |
| Hybrid + router | 0.934 | 0.900 | 0.961 | 19.9 ms |
| Hybrid (BM25 weight 0.3) + router | 0.954 | 0.928 | 0.969 | 19.5 ms |
| **Dense + router (default)** | **0.964** | **0.932** | **0.978** | **7.0 ms** |

**Finding 2, a negative result:** textbook equal-weight Reciprocal Rank Fusion *hurts*. It loses 8 points against dense alone, because BM25 is a much weaker ranker here and RRF gives it an equal vote. Down-weighting BM25 recovers most of the loss, but never beats dense.

**Finding 3:** the router adds **+4.8 to +6.4 points** of R@1 at negligible cost. Dense + router is best on both query sets at a third of hybrid's latency, so it is the production default. Weighted hybrid remains available to the agent as an opt-in `exact_terms=true` flag for queries that hinge on literal tokens (gene symbols, drug codes).

### 3.5 HNSW tuning in Chroma, on real queries (finding 5)
The usual way to benchmark an ANN index is to perturb in-corpus vectors and compare against exact search. We did that first. It reported **0.994** recall@10 at our original setting (M = 32, ef_search = 128).

Then the failure analysis showed true nearest neighbours being skipped. The 1,000 **real question embeddings** land *between* MedQuAD's tight clusters of near-duplicate NIH boilerplate, where greedy HNSW search stalls. On those queries the same index reaches only **0.952**, and 16 questions lose their true top-1. We rebuilt the sweep to report both query sets (`results/hnsw_benchmark.md`; 36,466 × 384 vectors; each point is a fresh build):

| M | ef_constr. | ef_search | R@10 perturbed | **R@10 real** | top-1 lost /1000 | p50 |
|---|---|---|---|---|---|---|
| 16 | 200 | 128 | 0.991 | 0.938 | 21 | 1.19 ms |
| 32 | 200 | 128 | 0.987 | 0.952 | 16 | 1.21 ms |
| 32 | 200 | 512 | 0.998 | 0.975 | 2 | 2.04 ms |
| 48 | 400 | 256 | 0.999 | 0.988 | 4 | 1.58 ms |
| 64 | 400 | 256 | 0.997 | 0.988 | 1 | 1.73 ms |
| **64** | **400** | **512** | 0.999 | **0.992** | **0** | **2.13 ms** |
| exact (numpy brute force) | – | – | 1 | 1 | 0 | 0.64 ms |

**Production setting:** `max_neighbors (M) = 64`, `ef_construction = 400`, `ef_search = 512`. It never loses the true top-1 on our 1,000 real queries, at about 2 ms. Changing M means rebuilding the graph, but not re-embedding: `scripts/compact_db.py` rebuilds the index from the stored vectors in about a minute.

Engineering notes from the sweep:
- **Brute force is faster at this scale.** Exact numpy search over 36k vectors takes 0.64 ms, against Chroma HNSW's 1–3 ms (which includes client overhead). The index pays off as the corpus grows: brute force is O(N). We still keep HNSW, since the deployment target is institution-scale corpora.
- **`collection.modify(ef_search=…)` only takes effect on reopen.** In Chroma 1.5 the new value is persisted immediately, but it does not change an index already loaded in the same process. Our sweep therefore builds each point fresh.
- **Builds aren't deterministic.** HNSW insertion is multi-threaded, so recall varies by about ±1 point between identical builds at low ef. We report single builds and favour settings with margin.
- **`delete_collection()` leaves orphaned segment folders** and an un-vacuumed SQLite file. A full rebuild now starts from an empty directory, and `compact_db.py` reclaims space (348 → 289 MB).
- **Large `get()` calls fail.** An unpaged `collection.get()` over about 35k rows exceeds SQLite's bound-variable limit, so all bulk reads go through a paged `get_all()`.

### 3.6 Full production index

| | |
|---|---|
| Chunks | 36,466 (`medical_kb`) + 300 (`patient_notes`) |
| Build (4-core CPU) | chunk 42 s · embed 865 s · index 38 s |
| On disk | 289 MB (vectors + HNSW + documents + metadata) |
| **End-to-end eval, canonical** (500 Q, production tool path, airgapped) | **R@1 0.932 · R@5 0.996 · R@10 0.998 · MRR 0.961** |
| **End-to-end eval, paraphrase** | **R@1 0.876 · R@5 0.984 · R@10 0.996 · MRR 0.925** |
| Tool-call latency | p50 ≈ 24 ms, p95 ≈ 29 ms (query embedding dominates; ANN ≈ 2 ms) |
| Remaining misses at k = 10 | 1 / 500 canonical, 2 / 500 paraphrase |

## 4. Yankee AI integration

```
Yankee AI agent (local LLM)
   │  MCP over stdio (no listening port)       or    HTTP, OpenAI tool schema, 127.0.0.1 only
   ▼
SafeHarborTools ── search_medical_knowledge(query, k, section?, exact_terms?)
                ── search_patient_notes(query, k, condition?)
                ── get_document(doc_id)
                ── index_stats()
   ▼
Retriever (dense + section router | weighted hybrid) ──► Chroma PersistentClient (HNSW cosine, M=64)
                    process runs under airgap(): any non-loopback socket raises
```

Integration uses standard agent-tool protocols, so nothing in Yankee AI needs to change:

- **MCP:** register `python server/mcp_server.py` as a stdio tool server (config snippet in the file header).
- **HTTP:** `GET /v1/tools` returns OpenAI function-calling schemas, and `POST /v1/tools/{name}` executes a tool.

Every result carries `doc_id`, `chunk_id`, `topic`, `section`, `source` and `matched_by`, so the agent can cite its evidence.

**Grounding check.** In a multi-step workflow, an agent often asks for a section that doesn't exist for a given disease: MedQuAD has only 46 "complications" pages. Nearest-neighbour search always returns *something*. In our first demo it cited "Neurological Complications of AIDS" as a complication of migraine. The reference agent now cites only evidence whose topic matches the condition, and otherwise reports *"not covered by the local knowledge base"*. For clinical use, refusing to cite is the correct behaviour.

**Demo workflow (video Part 2).** *"Brief me on our atrial-fibrillation patients and current treatment guidance."*

1. `search_patient_notes` returns 3 de-identified notes.
2. The agent extracts each note's conditions.
3. For each condition it calls `search_medical_knowledge("treatment of …")` and `search_medical_knowledge("complications of …")`, then applies the grounding check.
4. It composes a cited brief using `[PERSON_n]` placeholders.
5. Optionally, the authorised viewer re-identifies the brief locally from the vault (`--reidentify`).

Every tool call completes in tens of milliseconds, with the network refused. `agent/local_agent.py` runs this chain either deterministically or as a real tool-calling loop against a local Ollama model.

## 5. Privacy evaluation (`results/privacy_audit.json`)

| Check | Result |
|---|---|
| PHI identifiers in the stored documents + metadata (8 types, 2,099 unique) | **0 leaked** |
| Bare given names / surnames anywhere in the index | 0 |
| Outbound connection attempted inside `airgap()` | blocked (`EgressBlocked`) |
| Vault re-identification round-trip | 300 / 300 byte-exact |
| Chroma anonymized telemetry | off |
| Published dataset contains the vault | no (`export_db.sh` excludes it) |

## 6. Limitations
- **The evaluation queries are not clinician-written.** The canonical set uses NIH's own question templates. The paraphrase set uses template rewrites that avoid the NIH wording but still name the disease. A clinician-written test set is the most important next step.
- **The section router is regex-based.** It covers the 15 MedQuAD sections in English. A learned intent classifier would generalise better to other corpora.
- **Redaction is rule-based.** It reaches 0 leaks on synthetic notes whose PHI formats we know. Real notes have more variety: nicknames, free-text dates, rare names without cues. In production we would add a local de-identification NER model and keep the audit as a regression gate.
- **Synthetic notes** test the privacy pipeline, not clinical retrieval quality on real EHR text.
- **Single machine, single runs.** Benchmarks ran on a 4-core cloud VM. Latency includes noise from co-running jobs, and HNSW builds are non-deterministic (§3.5).

## 7. Impact beyond healthcare
The pattern is: de-identify before embedding, keep the vault separate, airgap the query path, expose only pseudonymised tools, and refuse to cite off-topic evidence. None of it is specific to medicine. Swapping the redactor's identifier classes adapts it to privileged legal files (party names, docket numbers), financial records (account and card numbers) and student data (FERPA).

## 8. Reproducibility
Everything runs on a CPU laptop with no GPU and no API keys:
```
python scripts/ingest.py            # full build ≈ 16 min
python scripts/benchmark.py         # A embedding models, B chunking, C retrieval modes
python scripts/benchmark_hnsw.py    # D HNSW (perturbed + real queries)
python scripts/evaluate_full.py     # F end-to-end on the full index + failure analysis
python scripts/audit_privacy.py     # E privacy audit
pytest -q
```
All randomness is seeded. The model, chunking and HNSW parameters are recorded in `data/chroma/manifest.json`.

## References
- Ben Abacha, A., & Demner-Fushman, D. (2019). A question-entailment approach to question answering. *BMC Bioinformatics* 20, 511.
- Morris, J. X., Kuleshov, V., Shmatikov, V., & Rush, A. M. (2023). Text embeddings reveal (almost) as much as text. *EMNLP 2023*.
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR 2009*.
- Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE TPAMI* 42(4).
- Wang, L., et al. (2022). Text embeddings by weakly-supervised contrastive pre-training. *arXiv:2212.03533*.
- U.S. Department of Health & Human Services. Guidance regarding methods for de-identification of protected health information (Safe Harbor, 45 CFR §164.514(b)(2)).
