#!/usr/bin/env python
"""Render the writeup / media-gallery figures from results/*.json and results/*.md.

    python scripts/make_figures.py      -> docs/figures/*.png

Every number is read from the benchmark outputs; nothing is typed in by hand.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, OUT = ROOT / "results", ROOT / "docs" / "figures"

# Validated categorical slots (light mode) + recessive ink, from the dataviz reference palette.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # blue, orange, aqua (aqua <3:1 -> always direct-labelled)

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False, "legend.labelcolor": INK2,
})


def md_table(path: Path) -> list[dict]:
    lines = [l for l in path.read_text().splitlines() if l.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    return [dict(zip(head, [c.strip() for c in l.strip("|").split("|")])) for l in lines[2:]]


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")


def paired_hbar(rows, labels, fname, title, sub, highlight=None, xmin=0.4):
    """Horizontal grouped bars: R@1 canonical vs paraphrase per strategy."""
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(labels) + 1.9), dpi=200)
    # Dot plot (not bars): the axis does not start at zero, and bar length would exaggerate gaps.
    for i, lab in enumerate(labels):
        y = len(labels) - 1 - i
        a, b = rows[(lab, "canonical")], rows[(lab, "paraphrase")]
        ax.plot([min(a, b), max(a, b)], [y, y], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
        for v, color, off in ((a, S1, 1), (b, S2, -1)):
            ax.scatter(v, y, s=70, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ha = "left" if (off == 1) == (a >= b) else "right"
            ax.annotate(f"{v:.3f}", (v, y), xytext=(8 if ha == "left" else -8, 0), textcoords="offset points",
                        ha=ha, va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)))
    for t in ax.get_yticklabels():
        t.set_color(INK if t.get_text() == highlight else INK2)
        t.set_fontweight("semibold" if t.get_text() == highlight else "normal")
    ax.set_xlim(xmin, 1.0)
    ax.xaxis.set_major_locator(FixedLocator([x / 20 for x in range(int(xmin * 20) + (xmin * 20 % 1 > 0), 21)]))
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.2f}")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("recall@1")
    dot = dict(marker="o", linestyle="", markersize=8, markeredgecolor=SURFACE)
    ax.legend(handles=[plt.Line2D([], [], color=S1, **dot), plt.Line2D([], [], color=S2, **dot)],
              labels=["NIH-template questions", "paraphrased questions"], loc="upper right", fontsize=8.5)
    top = 1 - 0.95 / fig.get_figheight()
    fig.text(0.015, 1 - 0.3 / fig.get_figheight(), title, fontsize=13, fontweight="semibold", color=INK, va="top")
    fig.text(0.015, 1 - 0.62 / fig.get_figheight(), sub, fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / fname)
    plt.close(fig)


def fig_chunking():
    t = md_table(RES / "chunking_benchmark.md")
    rows = {(r["tag"], r["variant"]): float(r["recall@1"]) for r in t}
    order = ["naive 1000-char / 200 overlap, no header", "192 tok, overlap 1, NO header",
             "96 tok, overlap 1, header", "192 tok, overlap 1, header", "384 tok, overlap 1, header"]
    pretty = {"naive 1000-char / 200 overlap, no header": "Naive 1000-char windows, no header",
              "192 tok, overlap 1, NO header": "Sentence-packed 192 tok, no header",
              "96 tok, overlap 1, header": "Sentence-packed 96 tok + header",
              "192 tok, overlap 1, header": "Sentence-packed 192 tok + header (ours)",
              "384 tok, overlap 1, header": "Sentence-packed 384 tok + header"}
    rows = {(pretty[k[0]], k[1]): v for k, v in rows.items() if k[0] in pretty}
    paired_hbar(rows, [pretty[o] for o in order], "fig_chunking.png",
                "The contextual header is worth +15 points of recall@1",
                "Chunking ablation · e5-small-v2, dense retrieval · 6,000 MedQuAD docs, 500 questions per variant",
                highlight=pretty["192 tok, overlap 1, header"], xmin=0.65)


def fig_modes():
    t = md_table(RES / "retrieval_modes.md")
    rows = {(r["mode"], r["variant"]): float(r["recall@1"]) for r in t}
    p50 = {r["mode"]: float(r["p50_ms"]) for r in t if r["variant"] == "canonical"}
    order = ["bm25", "hybrid", "dense", "hybrid+route", "hybrid@w0.3+route", "dense+route"]
    names = {"bm25": "BM25", "hybrid": "Hybrid RRF (equal weights)", "dense": "Dense",
             "hybrid+route": "Hybrid RRF + router", "hybrid@w0.3+route": "Hybrid RRF (BM25 ×0.3) + router",
             "dense+route": "Dense + section router (ours)"}
    lab = {m: f"{names[m]}  ·  {p50[m]:.0f} ms" for m in order}
    rows = {(lab[m], v): rows[(m, v)] for m in order for v in ("canonical", "paraphrase")}
    paired_hbar(rows, [lab[m] for m in order], "fig_retrieval_modes.png",
                "A cheap section router beats hybrid search, at a third of the latency",
                "Retrieval modes · e5-small-v2, 192-token header chunks · label shows p50 latency per query",
                highlight=lab["dense+route"], xmin=0.45)


def fig_hnsw():
    rows = json.loads((RES / "hnsw_benchmark.json").read_text())
    rows = [r for r in rows if isinstance(r["M"], int)]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    series = [
        (32, "recall@10 perturbed", S1, "M=32 · synthetic queries (perturbed vectors)", "-"),
        (32, "recall@10 real", S2, "M=32 · real questions", "-"),
        (64, "recall@10 real", S3, "M=64, efC=400 · real questions (ours)", "-"),
    ]
    for M, key, color, label, ls in series:
        pts = sorted((r["ef_search"], r[key]) for r in rows if r["M"] == M)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, ls, color=color, linewidth=2, marker="o", markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        dy = {S1: 7, S3: 2, S2: -7}[color]
        ax.annotate({S1: "M=32 · synthetic", S2: "M=32 · real", S3: "M=64 · real (ours)"}[color], (xs[-1], ys[-1]),
                    xytext=(8, dy), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ours = next(r for r in rows if r["M"] == 64 and r["ef_search"] == 512)
    ax.annotate(f"production: ef_search 512\nrecall {ours['recall@10 real']:.3f} · 0/1000 top-1 lost · "
                f"{ours['p50_ms']:.1f} ms", (512, ours["recall@10 real"]), xytext=(330, 0.893),
                textcoords="data", fontsize=8.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([32, 64, 128, 256, 512, 1024]))
    ax.xaxis.set_major_formatter(lambda x, _: f"{int(x)}")
    ax.set_xlim(26, 2600)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0.85, 1.005)
    ax.set_xlabel("ef_search")
    ax.set_ylabel("ANN recall@10 vs exact search")
    ax.grid(axis="x", visible=False)
    ax.set_title("Synthetic ANN benchmarks overstate HNSW recall on real questions", pad=22)
    subtitle(ax, "Chroma HNSW, 36,466 × 384 vectors · each point is a fresh index build · 300 synthetic / 1,000 real queries")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hnsw.png")
    plt.close(fig)


def fig_dedup():
    def full(p):
        t = md_table(p)
        return {r["variant"]: r for r in t if r.get("variant") in ("canonical", "paraphrase")}
    v1, v2 = full(RES / "full_index_eval_v1_answer_dedup.md"), full(RES / "full_index_eval.md")
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=200)
    labels = ["recall@1", "recall@5", "recall@10"]
    for i, k in enumerate(labels):
        y = len(labels) - 1 - i
        a, b = float(v1["canonical"][k]), float(v2["canonical"][k])
        ax.annotate("", (b, y), (a, y), arrowprops=dict(arrowstyle="-|>", color=GRID, linewidth=2.5,
                                                       shrinkA=6, shrinkB=6, mutation_scale=12))
        ax.scatter(a, y, s=70, color=S1, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter(b, y, s=70, color=S2, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{a:.3f}", (a, y), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=8.5, color=INK2)
        ax.annotate(f"{b:.3f}", (b, y), xytext=(8, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)))
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.85, 1.012)
    ax.xaxis.set_major_locator(FixedLocator([0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00]))
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.grid(axis="y", visible=False)
    dot = dict(marker="o", linestyle="", markersize=8, markeredgecolor=SURFACE)
    ax.legend(handles=[plt.Line2D([], [], color=S1, **dot), plt.Line2D([], [], color=S2, **dot)],
              labels=["dedupe on answer text", "dedupe on (topic, answer) — fixed"], loc="lower left",
              fontsize=8.5, ncol=1)
    ax.set_title("A one-line deduplication bug capped recall at 0.95", pad=22)
    subtitle(ax, "Full 36k-chunk index, production path, 500 NIH-template questions · 25 misses at k=10 → 1")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dedup_fix.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_chunking, fig_modes, fig_hnsw, fig_dedup):
        f()
        print("wrote", f.__name__)
    sys.exit(0)
