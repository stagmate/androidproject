# C. Retrieval modes (intfloat/e5-small-v2, 192-token header chunks)

| mode | variant | recall@1 | recall@5 | recall@10 | mrr@10 | p50_ms | p95_ms |
|---|---|---|---|---|---|---|---|
| dense | canonical | 0.882 | 0.958 | 0.960 | 0.916 | 6.03 | 7.78 |
| bm25 | canonical | 0.558 | 0.918 | 0.940 | 0.711 | 11.89 | 23.95 |
| hybrid | canonical | 0.800 | 0.948 | 0.960 | 0.868 | 20.27 | 33.12 |
| dense+route | canonical | 0.930 | 0.956 | 0.962 | 0.943 | 6.74 | 9.24 |
| hybrid+route | canonical | 0.896 | 0.958 | 0.962 | 0.923 | 19.82 | 32.39 |
| hybrid@w0.5+route | canonical | 0.906 | 0.956 | 0.960 | 0.929 | 20.37 | 33.84 |
| hybrid@w0.3+route | canonical | 0.916 | 0.954 | 0.958 | 0.933 | 18.86 | 28.66 |
| dense | paraphrase | 0.870 | 0.992 | 0.996 | 0.923 | 6.62 | 9.15 |
| bm25 | paraphrase | 0.510 | 0.932 | 0.970 | 0.694 | 11.05 | 19.42 |
| hybrid | paraphrase | 0.798 | 0.982 | 0.992 | 0.879 | 19.29 | 32.18 |
| dense+route | paraphrase | 0.930 | 0.994 | 0.996 | 0.959 | 6.85 | 9.08 |
| hybrid+route | paraphrase | 0.898 | 0.988 | 0.992 | 0.938 | 19.89 | 32.76 |
| hybrid@w0.5+route | paraphrase | 0.930 | 0.994 | 0.996 | 0.957 | 19.26 | 30.62 |
| hybrid@w0.3+route | paraphrase | 0.936 | 0.994 | 0.996 | 0.960 | 20.36 | 32.63 |
