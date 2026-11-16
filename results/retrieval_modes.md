# C. Retrieval modes (intfloat/e5-small-v2, 192-token header chunks)

| mode | variant | recall@1 | recall@5 | recall@10 | mrr@10 | p50_ms | p95_ms |
|---|---|---|---|---|---|---|---|
| dense | canonical | 0.884 | 0.960 | 0.962 | 0.918 | 5.98 | 9.03 |
| bm25 | canonical | 0.558 | 0.918 | 0.940 | 0.711 | 11.49 | 23.06 |
| hybrid | canonical | 0.802 | 0.950 | 0.960 | 0.870 | 19.76 | 32.11 |
| hybrid+route | canonical | 0.896 | 0.958 | 0.962 | 0.924 | 20.87 | 36.16 |
| dense | paraphrase | 0.870 | 0.992 | 0.996 | 0.923 | 6.17 | 9.1 |
| bm25 | paraphrase | 0.510 | 0.932 | 0.970 | 0.694 | 11.55 | 22.39 |
| hybrid | paraphrase | 0.798 | 0.982 | 0.992 | 0.879 | 25.78 | 39.22 |
| hybrid+route | paraphrase | 0.896 | 0.988 | 0.992 | 0.937 | 22.12 | 38.17 |
