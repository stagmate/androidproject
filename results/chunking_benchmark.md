# B. Chunking ablation (intfloat/e5-small-v2, dense)

| tag | variant | chunks | avg_tokens | recall@1 | recall@5 | recall@10 | mrr@10 |
|---|---|---|---|---|---|---|---|
| naive 1000-char / 200 overlap, no header | canonical | 11614 | 0.0 | 0.766 | 0.888 | 0.894 | 0.818 |
| naive 1000-char / 200 overlap, no header | paraphrase | 11614 | 0.0 | 0.730 | 0.884 | 0.896 | 0.802 |
| 96 tok, overlap 1, header | canonical | 31537 | 68.8 | 0.926 | 0.996 | 0.996 | 0.958 |
| 96 tok, overlap 1, header | paraphrase | 31537 | 68.8 | 0.878 | 0.996 | 0.996 | 0.929 |
| 192 tok, overlap 0, header | canonical | 12721 | 129.9 | 0.922 | 0.998 | 1.000 | 0.958 |
| 192 tok, overlap 0, header | paraphrase | 12721 | 129.9 | 0.888 | 0.994 | 0.998 | 0.938 |
| 192 tok, overlap 1, NO header | canonical | 12900 | 139.5 | 0.766 | 0.886 | 0.892 | 0.816 |
| 192 tok, overlap 1, NO header | paraphrase | 12900 | 139.5 | 0.710 | 0.860 | 0.880 | 0.781 |
| 192 tok, overlap 1, header | canonical | 13654 | 133.0 | 0.918 | 0.998 | 0.998 | 0.955 |
| 192 tok, overlap 1, header | paraphrase | 13654 | 133.0 | 0.870 | 0.994 | 0.996 | 0.927 |
| 384 tok, overlap 1, header | canonical | 8080 | 210.0 | 0.908 | 0.996 | 0.998 | 0.948 |
| 384 tok, overlap 1, header | paraphrase | 8080 | 210.0 | 0.880 | 0.992 | 0.998 | 0.930 |
