#!/usr/bin/env python3
"""Generate the paper figures (PDF) from a fixed-seed run.

    python scripts/make_figures.py [--seed 42] [--outdir ../paper/figs]

Figures:
    fig1_regularizer_sweep.pdf   Spearman & Pearson vs lambda1 (Sec. 5)
    fig2_recovery_scatter.pdf    recovered phi vs phi_true at lambda1=1 (Sec. 5)
    fig3_chain_dynamic_range.pdf Delta-phi vs lambda1 on a chain (Sec. 5)
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from poisson_inverse.synthetic import generate_funnel
from poisson_inverse import experiments as E

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
SOB = "#1b6ca8"   # blue  = Sobolev
TIK = "#c0392b"   # red   = Tikhonov


def _logx(vals):
    """Map lambda1 values (with a 0) to evenly spaced x positions."""
    return list(range(len(vals)))


def fig_sweep(funnel, outdir):
    res = E.regularizer_sweep(funnel)
    lam = res["lambdas"]
    x = _logx(lam)
    labels = ["0"] + [f"$10^{{{int(round(np.log10(l)))}}}$" for l in lam[1:]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, metric, title in [(axes[0], "spearman", "Spearman $\\rho(\\hat\\phi,\\phi_{\\mathrm{true}})$"),
                              (axes[1], "pearson", "Pearson $r(\\hat\\phi,\\phi_{\\mathrm{true}})$")]:
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
        ax.plot(x, [r[metric] for r in res["sobolev"]], "-o", color=SOB,
                label="graph-Sobolev", lw=2, ms=5)
        ax.plot(x, [r[metric] for r in res["tikhonov"]], "--s", color=TIK,
                label="Tikhonov", lw=2, ms=5)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel("regularization strength $\\lambda_1$")
        ax.set_title(title)
        ax.set_ylim(-0.65, 0.95)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, frameon=False, loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(outdir, "fig1_regularizer_sweep.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def fig_scatter(funnel, outdir):
    train = funnel.sessions[:int(0.8 * len(funnel.sessions))]
    pt = funnel.phi_true
    rs = E.fit_phi(funnel, lambda1=1.0, penalty="sobolev", sessions=train)
    sob = rs["phi"]
    tik = E.fit_phi(funnel, lambda1=1.0, penalty="tikhonov", sessions=train)["phi"]
    mask = E.scored_mask(funnel, train, rs["dag"])   # determined interior
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4), sharey=True)
    for ax, phi, color, name in [(axes[0], sob, SOB, "graph-Sobolev"),
                                 (axes[1], tik, TIK, "Tikhonov")]:
        ax.scatter(pt[mask], phi[mask], s=14, alpha=0.6, color=color,
                   edgecolors="none")
        ax.set_xlabel("$\\phi_{\\mathrm{true}}$ (planted absorption prob.)")
        ax.set_title(f"{name}  ($\\lambda_1=1$)")
    axes[0].set_ylabel("recovered $\\hat\\phi$")
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_recovery_scatter.pdf")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


def fig_chain(outdir):
    res = E.chain_dynamic_range(length=30,
                                lambdas=(0.0, 1e-2, 1e-1, 1.0, 10.0))
    lam = res["lambdas"]
    x = _logx(lam)
    labels = ["0"] + [f"$10^{{{int(round(np.log10(l)))}}}$" for l in lam[1:]]
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":", label="harmonic target")
    ax.plot(x, [r["delta_phi"] for r in res["sobolev"]], "-o", color=SOB,
            label="graph-Sobolev", lw=2, ms=5)
    ax.plot(x, [r["delta_phi"] for r in res["tikhonov"]], "--s", color=TIK,
            label="Tikhonov", lw=2, ms=5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("regularization strength $\\lambda_1$")
    ax.set_ylabel("interior dynamic range $\\Delta\\hat\\phi$")
    ax.set_title("Chain saturation ($L=30$)")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = os.path.join(outdir, "fig3_chain_dynamic_range.pdf")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "paper", "figs"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    funnel = generate_funnel(num_sessions=50_000, seed=args.seed)
    fig_sweep(funnel, args.outdir)
    fig_scatter(funnel, args.outdir)
    fig_chain(args.outdir)


if __name__ == "__main__":
    main()
