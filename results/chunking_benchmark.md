# B. Chunking ablation (intfloat/e5-small-v2, dense)

| tag | variant | chunks | avg_tokens | recall@1 | recall@5 | recall@10 | mrr@10 |
|---|---|---|---|---|---|---|---|
| naive 1000-char / 200 overlap, no header | canonical | 11532 | 0.0 | 0.758 | 0.894 | 0.898 | 0.815 |
| naive 1000-char / 200 overlap, no header | paraphrase | 11532 | 0.0 | 0.714 | 0.866 | 0.880 | 0.781 |
| 96 tok, overlap 1, header | canonical | 31510 | 68.9 | 0.888 | 0.956 | 0.958 | 0.919 |
| 96 tok, overlap 1, header | paraphrase | 31510 | 68.9 | 0.878 | 0.988 | 0.992 | 0.926 |
| 192 tok, overlap 0, header | canonical | 12670 | 130.2 | 0.888 | 0.960 | 0.962 | 0.920 |
| 192 tok, overlap 0, header | paraphrase | 12670 | 130.2 | 0.886 | 0.992 | 0.994 | 0.933 |
| 192 tok, overlap 1, NO header | canonical | 12831 | 139.9 | 0.744 | 0.888 | 0.894 | 0.808 |
| 192 tok, overlap 1, NO header | paraphrase | 12831 | 139.9 | 0.704 | 0.852 | 0.870 | 0.772 |
| 192 tok, overlap 1, header | canonical | 13556 | 133.5 | 0.882 | 0.958 | 0.960 | 0.916 |
| 192 tok, overlap 1, header | paraphrase | 13556 | 133.5 | 0.868 | 0.990 | 0.994 | 0.921 |
| 384 tok, overlap 1, header | canonical | 7998 | 211.6 | 0.866 | 0.954 | 0.958 | 0.906 |
| 384 tok, overlap 1, header | paraphrase | 7998 | 211.6 | 0.882 | 0.988 | 0.996 | 0.930 |
