"""
Experiments for the real corpora, where no ground-truth potential exists.
These replace the recovery-vs-truth experiments (E1/E5) with checks that are
meaningful without a planted field:

  R1  potential recovery     recovered phi per state (interpretability)
  R2  regularizer contrast   Sobolev vs Tikhonov at lambda1=1: do they
                             preserve the unregularized (lambda1=0) ordering,
                             and the interior dynamic range?
  R3  sink discovery         does the Poisson residual rank the *known*
                             conversion boundary at the top of all states?
  R4  bootstrap stability    resample sessions, refit, report per-state CV
  R5  downstream utility     is the recovered potential a usable node feature?
                             session-level conversion prediction (Sobolev vs
                             ridge), held-out ROC-AUC

All operate on a RealDataset (or any object exposing sessions, num_states,
sinks, conversion_sink_ids, abandon_sink_ids).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr

from .core import poisson_residual
from .experiments import fit_phi


def _interior_ids(ds) -> List[int]:
    sinks = set(ds.conversion_sink_ids) | set(ds.abandon_sink_ids) \
        | {k for k, _ in ds.sinks}
    return [i for i in range(ds.num_states) if i not in sinks]


# ---------------------------------------------------------------------------
# R1: potential recovery / interpretability
# ---------------------------------------------------------------------------
def recover_potential(ds, lambda1: float = 1.0, penalty: str = "sobolev",
                      rho: float = 2.0, tau: float = 0.7,
                      top_k: int = 10) -> Dict:
    """Fit phi and return a labeled, descending table of recovered values."""
    r = fit_phi(ds, lambda1=lambda1, penalty=penalty,
                rho=rho, tau=tau, top_k=top_k)
    phi = r["phi"]
    order = sorted(range(ds.num_states), key=lambda i: -phi[i])
    table = [{"state": i, "label": ds.state_labels.get(i, str(i)),
              "phi": float(phi[i])} for i in order]
    return {"dataset": ds.name, "lambda1": lambda1, "penalty": penalty,
            "num_dag_edges": len(r["dag"]), "phi_table": table}


# ---------------------------------------------------------------------------
# R2: regularizer contrast (no ground truth)
# ---------------------------------------------------------------------------
def regularizer_contrast(ds, lambda1: float = 1.0,
                         rho: float = 2.0, tau: float = 0.7,
                         top_k: int = 10) -> Dict:
    """Compare how Sobolev and Tikhonov at ``lambda1`` perturb the
    unregularized (lambda1=0) solution.

    Without ground truth we cannot score against truth, but we can ask which
    penalty *preserves* the structure of the base solve: its interior
    ordering (Spearman vs the lambda1=0 solve) and its interior dynamic
    range. Sobolev is expected to preserve both; Tikhonov to collapse them.
    """
    interior = _interior_ids(ds)
    base = fit_phi(ds, lambda1=0.0, penalty="none",
                   rho=rho, tau=tau, top_k=top_k)["phi"]
    out = {"dataset": ds.name, "lambda1": lambda1,
           "n_interior": len(interior)}
    if len(interior) < 2:
        out["note"] = "interior too small for rank comparison"
        return out

    def rng(phi):
        v = phi[interior]
        return float(v.max() - v.min())

    out["base_interior_range"] = rng(base)
    for penalty in ("sobolev", "tikhonov"):
        phi = fit_phi(ds, lambda1=lambda1, penalty=penalty,
                      rho=rho, tau=tau, top_k=top_k)["phi"]
        sp_ = float(spearmanr(base[interior], phi[interior]).statistic) \
            if np.ptp(base[interior]) > 0 and np.ptp(phi[interior]) > 0 \
            else float("nan")
        out[penalty] = {"spearman_vs_base": sp_,
                        "interior_range": rng(phi),
                        "range_retained": rng(phi) / (rng(base) + 1e-12)}
    return out


# ---------------------------------------------------------------------------
# R3: sink discovery against the known boundary
# ---------------------------------------------------------------------------
def sink_discovery_real(ds, lambda1: float = 1.0,
                        rho: float = 2.0, tau: float = 0.7,
                        top_k: int = 10) -> Dict:
    """Rank the known conversion sink(s) among all states by Poisson residual
    versus by recovered potential. On real data the boundary is known, so
    this validates the residual diagnostic rather than discovering anything.
    """
    r = fit_phi(ds, lambda1=lambda1, penalty="sobolev",
                rho=rho, tau=tau, top_k=top_k)
    phi, dag, b = r["phi"], r["dag"], r["b"]
    res = poisson_residual(dag, ds.num_states, phi, b)

    def ranks(score):
        order = np.argsort(-score).tolist()
        return [order.index(s) + 1 for s in ds.conversion_sink_ids]

    rr, pr = ranks(res), ranks(phi)
    return {"dataset": ds.name, "num_states": ds.num_states,
            "conversion_sinks": list(ds.conversion_sink_ids),
            "residual_ranks": rr, "residual_worst": int(max(rr)),
            "phi_ranks": pr, "phi_worst": int(max(pr))}


# ---------------------------------------------------------------------------
# R4: bootstrap stability
# ---------------------------------------------------------------------------
def bootstrap_stability(ds, n_boot: int = 20, lambda1: float = 1.0,
                        rho: float = 2.0, tau: float = 0.7, top_k: int = 10,
                        seed: int = 0) -> Dict:
    """Resample sessions with replacement, refit Sobolev phi, and report the
    per-state mean and coefficient of variation (std / |mean|)."""
    rng = np.random.default_rng(seed)
    n = len(ds.sessions)
    cols = []
    base = fit_phi(ds, lambda1=lambda1, penalty="sobolev",
                   rho=rho, tau=tau, top_k=top_k)["phi"]
    cols.append(base)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = [ds.sessions[i] for i in idx]
        phi = fit_phi(ds, lambda1=lambda1, penalty="sobolev",
                      rho=rho, tau=tau, top_k=top_k, sessions=boot)["phi"]
        cols.append(phi)
    M = np.vstack(cols)                       # (1 + n_boot, num_states)
    mean = M.mean(axis=0)
    std = M.std(axis=0)
    cv = std / (np.abs(mean) + 1e-12)
    interior = _interior_ids(ds)
    rows = sorted(interior, key=lambda i: -base[i])
    table = [{"state": i, "label": ds.state_labels.get(i, str(i)),
              "phi_mean": float(mean[i]), "phi_std": float(std[i]),
              "cv": float(cv[i])} for i in rows]
    return {"dataset": ds.name, "n_boot": n_boot, "stability_table": table}


# ---------------------------------------------------------------------------
# R5: downstream utility -- the potential as a node feature
# ---------------------------------------------------------------------------
def conversion_prediction(ds, lambda1: float = 1.0, rho: float = 2.0,
                          tau: float = 0.7, top_k: int = 10,
                          seeds=(0, 1, 2), train_frac: float = 0.7) -> Dict:
    """Is the recovered potential a usable node feature? Predict session-level
    conversion from the potentials of the interior (non-sink) states a session
    visits. The task is leakage-guarded: the label is the terminal sink, and
    features never include any sink state (whose potential is the pinned label).

    Potentials are fit on the training sessions only, then used to featurize
    both splits. We compare, by held-out ROC-AUC (averaged over ``seeds``):
    raw session statistics; raw + graph-Sobolev potential; raw + Tikhonov
    potential; each potential alone; and a bag-of-visited-states ceiling. The
    contrast isolates the penalty: same 4-D featurization, Sobolev vs ridge.
    """
    import statistics as st
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    conv = set(ds.conversion_sink_ids)
    sinks = conv | set(ds.abandon_sink_ids) | {k for k, _ in ds.sinks}
    N = ds.num_states

    def interior(s):
        return [u for u in s if u not in sinks]

    def label(s):
        return 1 if (s and s[-1] in conv) else 0

    def feats(s, phi=None):
        iv = interior(s)
        raw = [len(iv), len(set(iv))]
        if phi is None:
            return raw
        if not iv:
            return raw + [0.0, 0.0, 0.0, 0.0]
        p = np.array([phi[u] for u in iv])
        return raw + [float(p.mean()), float(p.max()),
                      float(p.min()), float(p[-1])]

    def bag(s):
        v = np.zeros(N)
        for u in interior(s):
            v[u] += 1.0
        return v

    accum = {k: [] for k in ("raw", "raw+sobolev", "raw+ridge",
                             "sobolev", "ridge", "bag_of_states")}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(ds.sessions))
        cut = int(train_frac * len(idx))
        tr, te = idx[:cut], idx[cut:]
        train = [ds.sessions[i] for i in tr]
        phi_s = fit_phi(ds, lambda1=lambda1, penalty="sobolev",
                        rho=rho, tau=tau, top_k=top_k, sessions=train)["phi"]
        phi_r = fit_phi(ds, lambda1=lambda1, penalty="tikhonov",
                        rho=rho, tau=tau, top_k=top_k, sessions=train)["phi"]
        ytr = np.array([label(ds.sessions[i]) for i in tr])
        yte = np.array([label(ds.sessions[i]) for i in te])

        def auc(fn):
            Xtr = np.array([fn(ds.sessions[i]) for i in tr])
            Xte = np.array([fn(ds.sessions[i]) for i in te])
            m = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
            return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))

        accum["raw"].append(auc(lambda s: feats(s)))
        accum["raw+sobolev"].append(auc(lambda s: feats(s, phi_s)))
        accum["raw+ridge"].append(auc(lambda s: feats(s, phi_r)))
        accum["sobolev"].append(auc(lambda s: feats(s, phi_s)[2:]))
        accum["ridge"].append(auc(lambda s: feats(s, phi_r)[2:]))
        accum["bag_of_states"].append(auc(bag))

    summary = {k: {"auc_mean": st.mean(v), "auc_std": st.pstdev(v)}
               for k, v in accum.items()}
    return {"dataset": ds.name, "num_states": N, "n_sessions": len(ds.sessions),
            "conversion_rate": float(np.mean([label(s) for s in ds.sessions])),
            "seeds": list(seeds), "auc": summary}
