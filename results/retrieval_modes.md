# C. Retrieval modes (intfloat/e5-small-v2, 192-token header chunks)

| mode | variant | recall@1 | recall@5 | recall@10 | mrr@10 | p50_ms | p95_ms |
|---|---|---|---|---|---|---|---|
| dense | canonical | 0.916 | 0.996 | 0.996 | 0.953 | 6.85 | 10.57 |
| bm25 | canonical | 0.620 | 0.956 | 0.982 | 0.764 | 10.87 | 20.39 |
| hybrid | canonical | 0.836 | 0.992 | 0.998 | 0.905 | 20.02 | 30.84 |
| dense+route | canonical | 0.964 | 0.996 | 0.996 | 0.978 | 6.97 | 10.58 |
| hybrid+route | canonical | 0.934 | 0.996 | 0.998 | 0.961 | 19.86 | 30.33 |
| hybrid@w0.5+route | canonical | 0.948 | 0.992 | 0.992 | 0.967 | 19.64 | 29.95 |
| hybrid@w0.3+route | canonical | 0.954 | 0.990 | 0.994 | 0.969 | 19.5 | 28.63 |
| dense | paraphrase | 0.868 | 0.992 | 0.994 | 0.925 | 6.67 | 8.83 |
| bm25 | paraphrase | 0.542 | 0.936 | 0.974 | 0.709 | 12.08 | 20.16 |
| hybrid | paraphrase | 0.792 | 0.988 | 0.994 | 0.878 | 21.1 | 30.19 |
| dense+route | paraphrase | 0.932 | 0.992 | 0.994 | 0.959 | 6.64 | 9.43 |
| hybrid+route | paraphrase | 0.900 | 0.992 | 0.992 | 0.940 | 20.08 | 28.77 |
| hybrid@w0.5+route | paraphrase | 0.926 | 0.990 | 0.992 | 0.953 | 21.37 | 30.98 |
| hybrid@w0.3+route | paraphrase | 0.928 | 0.988 | 0.992 | 0.955 | 20.99 | 30.65 |
