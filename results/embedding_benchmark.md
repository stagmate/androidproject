# A. Embedding models (dense only, 192-token header chunks, 6000 docs, 500 queries)

| tag | variant | recall@1 | recall@5 | recall@10 | mrr@10 | embed_chunks_per_s | p50_ms |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | canonical | 0.810 | 0.988 | 0.996 | 0.890 | 46.6 | 6.64 |
| all-MiniLM-L6-v2 | paraphrase | 0.746 | 0.990 | 1.000 | 0.856 | 46.6 | 6.68 |
| multi-qa-MiniLM-L6-cos-v1 | canonical | 0.842 | 0.990 | 0.994 | 0.908 | 81.1 | 6.71 |
| multi-qa-MiniLM-L6-cos-v1 | paraphrase | 0.762 | 0.974 | 0.986 | 0.857 | 81.1 | 6.75 |
| bge-small-en-v1.5 | canonical | 0.894 | 0.994 | 0.998 | 0.940 | 42.2 | 6.53 |
| bge-small-en-v1.5 | paraphrase | 0.848 | 0.990 | 0.994 | 0.911 | 42.2 | 6.39 |
| e5-small-v2 | canonical | 0.920 | 1.000 | 1.000 | 0.957 | 42.3 | 6.56 |
| e5-small-v2 | paraphrase | 0.870 | 0.992 | 0.994 | 0.926 | 42.3 | 6.32 |
