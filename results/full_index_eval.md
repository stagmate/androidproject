# F. Full-index evaluation (production path: e5-small-v2, dense + section router, airgapped)

Index: {'patient_notes': 300, 'medical_kb': 36466} | queries: 500 per variant

| variant | recall@1 | recall@5 | recall@10 | mrr@10 | p50 ms (tool call) | p95 ms |
|---|---|---|---|---|---|---|
| canonical | 0.932 | 0.996 | 0.998 | 0.961 | 23.91 | 29.38 |
| paraphrase | 0.876 | 0.984 | 0.996 | 0.925 | 23.23 | 28.87 |

## Misses at k=10

**canonical** (1): wrong topic: 1

**paraphrase** (2): wrong topic: 2

Examples:

- *do you have information about blood count tests* (gold: Blood Count Tests / information) → top: ('Hemophilia', 'exams and tests') — wrong topic
- *ongoing clinical trials for Chronic Pain* (gold: Chronic Pain / research) → top: ('Back Pain', 'research') — wrong topic
- *explain Blood Count Tests* (gold: Blood Count Tests / information) → top: ('Nasopharyngeal Cancer', 'exams and tests') — wrong topic
