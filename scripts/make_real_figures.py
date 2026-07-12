#!/usr/bin/env python3
"""Generate the real-data figures (PDF) from results_<dataset>.json.

    python scripts/make_real_figures.py [--results_dir results] [--outdir ../paper/figs]

Figures:
    fig4_real_regularizer_contrast.pdf   range retained + rank agreement (Sec. 8)
    fig5_real_potentials.pdf             recovered phi: RetailRocket + Trivago
    fig6_otto_event_separation.pdf       OTTO clicks vs carts potential
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
SOB = "#1b6ca8"
TIK = "#c0392b"
SINKC = "#7f8c8d"


def load_results(results_dir):
    out = {}
    for d in ("retailrocket", "trivago", "otto"):
        p = os.path.join(results_dir, f"results_{d}.json")
        if os.path.exists(p):
            out[d] = json.load(open(p))
    return out


def fig_contrast(res, outdir):
    order = [k for k in ("retailrocket", "trivago", "otto") if k in res]
    names = {"retailrocket": "RetailRocket", "trivago": "Trivago", "otto": "OTTO"}
    labels, sob_rng, tik_rng, sob_sp, tik_sp = [], [], [], [], []
    for k in order:
        r2 = res[k]["R2_regularizer_contrast"]
        ns = res[k]["R3_sink_discovery"]["num_states"]
        labels.append(f"{names[k]}\n({ns} states)")
        sob_rng.append(r2["sobolev"]["range_retained"])
        tik_rng.append(r2["tikhonov"]["range_retained"])
        sob_sp.append(r2["sobolev"]["spearman_vs_base"])
        tik_sp.append(r2["tikhonov"]["spearman_vs_base"])

    x = np.arange(len(order)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax = axes[0]
    b1 = ax.bar(x - w/2, sob_rng, w, color=SOB, label="graph-Sobolev")
    b2 = ax.bar(x + w/2, tik_rng, w, color=TIK, label="Tikhonov")
    ax.set_title("Interior dynamic range retained\n(fraction of unregularized range)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, max(sob_rng) * 1.28)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width()/2, h),
                        ha="center", va="bottom", fontsize=8)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    ax.bar(x - w/2, sob_sp, w, color=SOB, label="graph-Sobolev")
    ax.bar(x + w/2, tik_sp, w, color=TIK, label="Tikhonov")
    ax.set_title("Rank agreement with the\nunregularized solve (Spearman)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
    fig.tight_layout()
    p = os.path.join(outdir, "fig4_real_regularizer_contrast.pdf")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


def _phi_bars(ax, table, title, sink_words=("SINK",)):
    table = sorted(table, key=lambda r: r["phi"])
    labels = [r["label"] for r in table]
    vals = [r["phi"] for r in table]
    colors = [SINKC if any(w in r["label"] for w in sink_words) else SOB
              for r in table]
    y = np.arange(len(table))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel(r"recovered $\hat\phi$")
    ax.set_title(title)


def fig_potentials(res, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    _phi_bars(axes[0], res["retailrocket"]["R1_potential"]["phi_table"],
              "RetailRocket (4 states)")
    _phi_bars(axes[1], res["trivago"]["R1_potential"]["phi_table"],
              "Trivago (11 states)")
    fig.tight_layout()
    p = os.path.join(outdir, "fig5_real_potentials.pdf")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


def fig_otto(res, outdir):
    t = res["otto"]["R1_potential"]["phi_table"]
    clicks = [r["phi"] for r in t if r["label"].startswith("clicks")]
    carts = [r["phi"] for r in t if r["label"].startswith("carts")]
    sinks = {r["label"]: r["phi"] for r in t if r["label"].startswith("SINK")}
    fig, ax = plt.subplots(figsize=(5.8, 4))
    rng = np.random.default_rng(0)
    for i, (vals, col) in enumerate([(clicks, "#8e8e8e"), (carts, SOB)]):
        xs = i + 0.06 * rng.standard_normal(len(vals))
        ax.scatter(xs, vals, s=18, alpha=0.6, color=col, edgecolors="none")
        ax.scatter([i], [np.mean(vals)], marker="_", s=900, color="black",
                   zorder=5, linewidths=2)
    for name, phi in sinks.items():
        ax.axhline(phi, color=SINKC, lw=1, ls="--")
        ax.text(1.55, phi, name.replace("SINK:", ""), va="center",
                fontsize=8, color=SINKC)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["clicks states", "carts states"])
    ax.set_ylabel(r"recovered $\hat\phi$")
    ax.set_title("OTTO: recovered potential by event type\n(105-state multi-sink graph)")
    ax.set_xlim(-0.5, 2.0)
    fig.tight_layout()
    p = os.path.join(outdir, "fig6_otto_event_separation.pdf")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(
        os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "paper", "figs"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    res = load_results(args.results_dir)
    if not res:
        raise SystemExit(f"no results_*.json found in {args.results_dir}")
    fig_contrast(res, args.outdir)
    if "retailrocket" in res and "trivago" in res:
        fig_potentials(res, args.outdir)
    if "otto" in res:
        fig_otto(res, args.outdir)


if __name__ == "__main__":
    main()
