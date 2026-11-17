# Kaggle submission checklist

The Stage 1 pass/fail audit disqualifies a submission that misses any one item.

| Requirement | Status | Where / what to do |
|---|---|---|
| Kaggle Writeup with title, subtitle, team, track | text ready | Paste `WRITEUP.md`. Fill in the **Team** table (1–3 members, with status). Select the **Healthcare & Life Sciences** track. |
| Cover image | ready | `docs/figures/cover.png` (1120×560, 2:1) |
| Media gallery figures | ready | `docs/figures/architecture.png`, `fig_dedup_fix.png`, `fig_chunking.png`, `fig_retrieval_modes.png`, `fig_hnsw.png`. Upload them to the gallery. Kaggle does not resolve the repo-relative image links in the markdown, so re-insert each one through the editor's image button. |
| Public code repository | **make public** | https://github.com/stagmate/androidproject (consider renaming it, e.g. `safeharbor-rag`, and updating the link in `WRITEUP.md`). It must open without a login. |
| Vector DB + original data | **upload** | `bash scripts/export_db.sh` builds `dist/safeharbor-chroma-db.zip` (~149 MB). Publish it as a public Kaggle Dataset or Hugging Face dataset, then paste the URL into `WRITEUP.md` → Links and attach it under "Datasets". |
| YouTube video ≤ 10 min, Part 1 (vector DB) + Part 2 (Yankee AI demo) | **record** | Follow `docs/VIDEO_SCRIPT.md`. Set visibility to Public or Unlisted (viewable without login). Paste the link into `WRITEUP.md` and the media gallery. |
| Yankee AI integration shown in the demo | **verify** | Confirm Yankee AI can register an MCP stdio server or OpenAI-style HTTP tools (`server/`). If it uses another mechanism, adapt the thin wrapper around `safeharbor/tools.py`. |
| Submit | — | Press **Submit** (not just Save) before 31 Mar 2027, 20:29 IST. |

To regenerate the figures after re-running any benchmark:
```
python scripts/make_figures.py          # charts, read from results/
node scripts/render_html_figures.js     # cover + architecture (Playwright)
```
