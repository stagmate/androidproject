# A. Embedding models (dense only, 192-token header chunks, 6000 docs, 500 queries)

| tag | variant | recall@1 | recall@5 | recall@10 | mrr@10 | embed_chunks_per_s | p50_ms |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | canonical | 0.776 | 0.952 | 0.960 | 0.856 | 92.1 | 5.53 |
| all-MiniLM-L6-v2 | paraphrase | 0.742 | 0.988 | 0.998 | 0.849 | 92.1 | 4.75 |
| multi-qa-MiniLM-L6-cos-v1 | canonical | 0.802 | 0.952 | 0.958 | 0.871 | 96.0 | 4.76 |
| multi-qa-MiniLM-L6-cos-v1 | paraphrase | 0.750 | 0.976 | 0.992 | 0.852 | 96.0 | 5.37 |
| bge-small-en-v1.5 | canonical | 0.864 | 0.952 | 0.954 | 0.904 | 49.9 | 5.39 |
| bge-small-en-v1.5 | paraphrase | 0.840 | 0.986 | 0.992 | 0.906 | 49.9 | 5.57 |
| e5-small-v2 | canonical | 0.884 | 0.960 | 0.962 | 0.918 | 48.4 | 5.15 |
| e5-small-v2 | paraphrase | 0.870 | 0.992 | 0.996 | 0.923 | 48.4 | 5.17 |
