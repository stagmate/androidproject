# C. Retrieval modes (BAAI/bge-small-en-v1.5, 192-token header chunks)

| mode | variant | recall@1 | recall@5 | recall@10 | mrr@10 | p50_ms | p95_ms |
|---|---|---|---|---|---|---|---|
| dense | canonical | 1.000 | 1.000 | 1.000 | 1.000 | 4.73 | 5.14 |
| bm25 | canonical | 0.967 | 1.000 | 1.000 | 0.983 | 0.45 | 0.97 |
| hybrid | canonical | 1.000 | 1.000 | 1.000 | 1.000 | 5.31 | 5.86 |
| hybrid+route | canonical | 1.000 | 1.000 | 1.000 | 1.000 | 5.25 | 6.06 |
| dense | paraphrase | 1.000 | 1.000 | 1.000 | 1.000 | 5.18 | 6.86 |
| bm25 | paraphrase | 0.900 | 1.000 | 1.000 | 0.950 | 0.55 | 1.08 |
| hybrid | paraphrase | 1.000 | 1.000 | 1.000 | 1.000 | 5.71 | 6.16 |
| hybrid+route | paraphrase | 1.000 | 1.000 | 1.000 | 1.000 | 5.67 | 6.34 |
