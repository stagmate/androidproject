# Video script (target 8–9 min, hard limit 10:00)

Record at 1080p. Terminal font ≥ 18 pt. Turn Wi-Fi **off** on camera before Part 2. That's the
strongest privacy proof you can show.

---

## Cold open (0:00 – 0:40)
**Screen:** a synthetic discharge note full of PHI (`data/raw/synthetic_notes_with_phi.jsonl`).
> "This is a discharge summary. Name, birth date, record number, phone, home address. Under
> HIPAA, sending it to a cloud LLM is a reportable disclosure. But clinicians still want an
> agent that can read their notes and the medical literature together. SafeHarbor-RAG gives
> them that agent, and nothing ever leaves the laptop."

Title card: **SafeHarbor-RAG: Privacy-First Clinical Retrieval for Yankee AI on Chroma DB**. Team names.

---

## PART 1: The Vector Database (0:40 – 5:00)

**1. Data (0:40 – 1:30).** Show `results/` and the architecture diagram (`docs/architecture.md`).
- Public knowledge: MedQuAD, 16,407 NIH Q&A pairs from 12 NIH sites. After topic-aware deduplication there are 16,346 documents and 36,466 chunks.
- Private data: 300 synthetic clinical notes with realistic PHI. They stand in for an EHR export.
- Two collections: `medical_kb` and `patient_notes`.

**2. De-identification before embedding (1:30 – 2:20).** Run:
```
python -c "from safeharbor.data import generate_notes; from safeharbor.privacy import PHIRedactor; n=generate_notes(1)[0]; print(n['text']); print(PHIRedactor().redact(n['text'], n['_known_names']).text)"
```
> "Embeddings can be inverted, so we redact before we embed. Placeholders stay consistent within a note, which lets the agent still reason about '[PERSON_1]'. The mapping goes into a separate local SQLite vault."
Show `results/privacy_audit.json`: **0 leaked identifiers**.

**3. Chunking (2:20 – 3:10).** Show the table in `results/chunking_benchmark.md`.
- Chunks are packed from whole sentences, and size is counted with the embedding model's own tokenizer.
- Each chunk gets a contextual header: "Cystic Fibrosis | inheritance: …". Call out the recall gain from the header in the table.
- Point to the naive 1000-character baseline row.

**4. Embedding model selection (3:10 – 3:50).** Show `results/embedding_benchmark.md`. Four CPU models were compared, and e5-small-v2 was chosen: +11 points of R@1 over MiniLM.

**4b. The bug we found (30 s, judges love this).** Show `results/full_index_eval_v1_answer_dedup.md`. Deduplicating on answer text alone mislabelled the shared NIH boilerplate. Fixing it moved R@10 from 0.950 to 0.998.

**5. Hybrid retrieval + HNSW tuning (3:50 – 5:00).** Show `results/retrieval_modes.md` and `results/hnsw_benchmark.md`.
- Negative result: equal-weight RRF hybrid *loses* 8 points. The section router is the winner: dense + router reaches R@1 0.964.
- HNSW: synthetic benchmarks said 0.994 recall, but real questions got only 0.952. Tuning on real queries gives M=64, ef=512: 0.992 recall and 0 top-1 misses in about 2 ms.

---

## PART 2: Yankee AI Integration & Demo (5:00 – 9:00)

**6. Architecture (5:00 – 5:40).** Yankee AI agent → MCP stdio (no port) or loopback HTTP → SafeHarbor tools → Chroma (persistent, on disk). The airgap guard blocks any non-loopback socket.

**7. Live demo (5:40 – 8:30).** Wi-Fi off.
1. `python scripts/audit_privacy.py`: egress blocked, 0 leaks.
2. In Yankee AI, register the MCP server (show the config). Ask:
   *"Brief me on our atrial fibrillation patients and what current guidance says about treatment and complications."*
   Point out the tool calls: `search_patient_notes` → `search_medical_knowledge(section=treatment)` → … The cited answer keeps `[PERSON_1]` placeholders.
3. Fallback, if Yankee AI isn't available on the recording machine: `python agent/local_agent.py "…" --reidentify`. It runs the same tool chain. Re-identification happens only at the end, locally, for the authorised viewer.
4. Show the tool latencies printed by the agent (milliseconds).

**8. Close (8:30 – 9:00).**
> "Healthcare today, but the same pattern works for legal privilege, financial records and student data. High-performance agentic retrieval, and the data never leaves the room."
