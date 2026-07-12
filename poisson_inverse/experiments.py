"""
The experiments that support the paper, each returning plain dicts so the
driver can serialize them to JSON and the figure script can plot them.

  E1  regularizer sweep      Sobolev vs Tikhonov: Spearman/Pearson(phi, phi_true)
                             across lambda1   (Sec. 5, Fig. 1)
  E2  chain dynamic range    Sobolev vs Tikhonov: Delta-phi on a long chain
                             (Sec. 5, Fig. 2)
  E3  blind sink discovery   Poisson residual vs direct-phi ranking of sinks
                             (Sec. 6)
  E4  nilpotency             ||A^{L+1}||_F = 0 exactly on the extracted DAG
                             (Sec. 3)
  E5  gradient preservation  weighted cosine(recovered grad, planted grad)
                             (Sec. 3)
  E6  extraction ablation    Sobolev vs Tikhonov under HHD vs topological-sort
                             supports   (Sec. 7.2)
  E7  regularizer robustness contrast margin across seeds and instrument
                             configurations   (Sec. 7.3)
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .core import (
    empirical_flow, dominance_filter, hhd_dag_extract, top_k_prune,
    divergence, solve_poisson, poisson_residual, nilpotency_index,
    topological_dag, reachable_to_boundary,
)
from .synthetic import SyntheticFunnel, generate_funnel, build_chain


# ---------------------------------------------------------------------------
# Fit pipeline
# ---------------------------------------------------------------------------
def fit_phi(funnel: SyntheticFunnel, lambda1: float = 0.0,
            penalty: str = "sobolev", rho: float = 2.0, tau: float = 0.7,
            top_k: int = 10, sessions: Optional[List[List[int]]] = None) -> Dict:
    """Run flow -> dominance -> HHD-DAG -> divergence -> Poisson solve."""
    sess = funnel.sessions if sessions is None else sessions
    N = funnel.num_states
    F = empirical_flow(sess, N)
    F_dom = dominance_filter(F, rho)
    phi0, retained, edge_grad = hhd_dag_extract(F, F_dom, N, tau=tau)
    dag = top_k_prune(retained, top_k)
    b = divergence(F, N)
    sink_map = dict(funnel.sinks)
    phi, residual = solve_poisson(dag, N, sink_map, b,
                                  lambda1=lambda1, penalty=penalty)
    return {"phi": phi, "phi0": phi0, "dag": dag, "b": b,
            "edge_grad": edge_grad, "residual": residual, "F": F}


def well_visited_mask(funnel: SyntheticFunnel, sessions: List[List[int]],
                      floor: int = 100) -> np.ndarray:
    """Interior states with >= floor training visits (the honest slice)."""
    sinks = set(funnel.conversion_sink_ids) | set(funnel.abandon_sink_ids)
    visits = Counter(u for s in sessions for u in s)
    return np.array([(visits[i] >= floor) and (i not in sinks)
                     for i in range(funnel.num_states)])


def scored_mask(funnel: SyntheticFunnel, sessions: List[List[int]],
                dag) -> np.ndarray:
    """Well-visited interior states that are also *determined*: they have a
    directed path to the boundary (Theorem 1). No-path nodes are undetermined
    by the Dirichlet data and are excluded from recovery scoring."""
    sinks = set(funnel.conversion_sink_ids) | set(funnel.abandon_sink_ids)
    reach = reachable_to_boundary(dag, funnel.num_states, sinks)
    return well_visited_mask(funnel, sessions) & reach


def _rank_corr(pt: np.ndarray, phi: np.ndarray, decimals: int = 6) -> float:
    """Tie-robust Spearman: round the estimate before ranking so the near-zero
    (saturated) cluster ties deterministically rather than being ordered by
    floating-point noise."""
    return float(spearmanr(pt, np.round(phi, decimals)).statistic)


def _train_split(funnel: SyntheticFunnel, frac: float = 0.8):
    n = int(len(funnel.sessions) * frac)
    return funnel.sessions[:n]


# ---------------------------------------------------------------------------
# E1: regularizer sweep
# ---------------------------------------------------------------------------
def _ndcg_at_k(pred: np.ndarray, rel: np.ndarray, k: int = 5) -> float:
    """NDCG@k of the ranking induced by ``pred`` against graded relevance
    ``rel`` (here the ground-truth potential phi_true, in [0,1])."""
    order = np.argsort(-pred)[:k]
    disc = np.log2(np.arange(2, 2 + len(order)))
    dcg = float(np.sum(rel[order] / disc))
    ideal = np.sort(rel)[::-1][:k]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, 2 + len(ideal)))))
    return dcg / idcg if idcg > 0 else 0.0


def regularizer_sweep(funnel: SyntheticFunnel,
                      lambdas=(0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
                      rho: float = 2.0, tau: float = 0.7) -> Dict:
    """Spearman/Pearson and top-k ranking power (NDCG@5, NDCG@10) of the
    recovered phi vs phi_true, for both penalties on the HHD-extracted DAG."""
    train = _train_split(funnel)
    pt = funnel.phi_true
    # Extraction is fixed across lambda/penalty; get the DAG once to build the
    # determined-interior scoring mask (well-visited AND path-to-boundary).
    dag0 = fit_phi(funnel, lambda1=0.0, penalty="sobolev",
                   rho=rho, tau=tau, sessions=train)["dag"]
    mask = scored_mask(funnel, train, dag0)
    rel = pt[mask]                                   # graded relevance = phi_true
    out = {"lambdas": list(lambdas), "n_well": int(mask.sum()),
           "sobolev": [], "tikhonov": []}
    for penalty in ("sobolev", "tikhonov"):
        for lam in lambdas:
            r = fit_phi(funnel, lambda1=lam, penalty=penalty,
                        rho=rho, tau=tau, sessions=train)
            phi = r["phi"]
            sp_ = _rank_corr(pt[mask], phi[mask])
            pe_ = float(pearsonr(pt[mask], phi[mask]).statistic)
            out[penalty].append({"lambda1": lam, "spearman": sp_, "pearson": pe_,
                                 "ndcg5": _ndcg_at_k(phi[mask], rel, 5),
                                 "ndcg10": _ndcg_at_k(phi[mask], rel, 10)})
    return out


# ---------------------------------------------------------------------------
# E2: dynamic range on a chain
# ---------------------------------------------------------------------------
def chain_dynamic_range(length: int = 30,
                        lambdas=(0.0, 1e-2, 1e-1, 1.0, 10.0)) -> Dict:
    """Delta-phi (max-min over interior) on a single chain, both penalties.

    The chain has zero interior divergence and Dirichlet boundaries
    {source=0, sink=1}; the harmonic solution is the linear ramp with
    Delta-phi = 1. A range-preserving regularizer keeps Delta-phi near 1.
    """
    dag, N, sink_map, b = build_chain(length)
    sink = N - 1
    interior = [i for i in range(N) if i != sink]   # source + chain (sink excluded)
    out = {"length": length, "lambdas": list(lambdas),
           "sobolev": [], "tikhonov": []}
    for penalty in ("sobolev", "tikhonov"):
        for lam in lambdas:
            phi, _ = solve_poisson(dag, N, sink_map, b,
                                   lambda1=lam, penalty=penalty)
            dphi = float(phi[interior].max() - phi[interior].min())
            out[penalty].append({"lambda1": lam, "delta_phi": dphi})
    return out


# ---------------------------------------------------------------------------
# E3: blind sink discovery
# ---------------------------------------------------------------------------
def sink_discovery(funnel: SyntheticFunnel, lambda1: float = 1.0,
                   rho: float = 2.0, tau: float = 0.7) -> Dict:
    """Rank the true conversion sinks by Poisson residual vs by direct phi."""
    train = _train_split(funnel)
    r = fit_phi(funnel, lambda1=lambda1, penalty="sobolev",
                rho=rho, tau=tau, sessions=train)
    phi, dag, b = r["phi"], r["dag"], r["b"]
    N = funnel.num_states
    res = poisson_residual(dag, N, phi, b)

    def ranks(score, descending=True):
        order = np.argsort(-score if descending else score).tolist()
        return [order.index(s) + 1 for s in funnel.conversion_sink_ids]

    res_ranks = ranks(res, descending=True)
    phi_ranks = ranks(phi, descending=True)
    return {"num_states": N, "num_sinks": len(funnel.conversion_sink_ids),
            "residual_ranks": res_ranks, "residual_worst": int(max(res_ranks)),
            "phi_ranks": phi_ranks, "phi_worst": int(max(phi_ranks))}


def sink_discovery_cross_regime(seed: int = 42,
                                abandon_rates=(0.05, 0.20, 0.50)) -> Dict:
    """Worst-case sink rank under the residual across abandon regimes."""
    rows = []
    for ar in abandon_rates:
        f = generate_funnel(abandon_rate=ar, seed=seed)
        d = sink_discovery(f)
        rows.append({"abandon_rate": ar, "residual_ranks": d["residual_ranks"],
                     "residual_worst": d["residual_worst"]})
    return {"rows": rows}


# ---------------------------------------------------------------------------
# E4: nilpotency
# ---------------------------------------------------------------------------
def nilpotency(funnel: SyntheticFunnel, rho: float = 2.0, tau: float = 0.7) -> Dict:
    train = _train_split(funnel)
    r = fit_phi(funnel, lambda1=1.0, penalty="sobolev",
                rho=rho, tau=tau, sessions=train)
    m, frob = nilpotency_index(r["dag"], funnel.num_states)
    return {"nilpotency_index": m, "frobenius_at_index": frob}


# ---------------------------------------------------------------------------
# E5: gradient-direction preservation
# ---------------------------------------------------------------------------
def gradient_preservation(funnel: SyntheticFunnel,
                          rho: float = 2.0, tau: float = 0.7) -> Dict:
    """Weighted cosine between recovered and planted gradient on the DAG.

    Scoped claim: HHD preserves gradient *direction*, not magnitude.
    """
    train = _train_split(funnel)
    r = fit_phi(funnel, lambda1=1.0, penalty="sobolev",
                rho=rho, tau=tau, sessions=train)
    phi, phi0, dag = r["phi"], r["phi0"], r["dag"]
    pt = funnel.phi_true
    if not dag:
        return {"cosine_hhd_gradient": float("nan"),
                "cosine_recovered_phi": float("nan"), "n_edges": 0}

    def wcos(field):
        gh = np.array([field[v] - field[u] for (u, v, _) in dag])
        gt = np.array([pt[v] - pt[u] for (u, v, _) in dag])
        w = np.array([w for (_, _, w) in dag])
        num = float(np.sum(w * gh * gt))
        den = float(np.sqrt(np.sum(w * gh * gh) * np.sum(w * gt * gt)))
        return num / (den + 1e-30)

    # Primary §3 diagnostic: the HHD gradient component phi0 in isolation
    # (before the Dirichlet/Sobolev solve). Secondary: the final recovered
    # potential's gradient direction.
    return {"cosine_hhd_gradient": wcos(phi0),
            "cosine_recovered_phi": wcos(phi),
            "n_edges": len(dag)}


# ---------------------------------------------------------------------------
# E6: extraction ablation -- does the Hodge projection matter?
# ---------------------------------------------------------------------------
def _mean_visit_position(sessions, num_states):
    """Cheap data-only downstreamness score: mean fractional session position."""
    s = np.zeros(num_states)
    c = np.zeros(num_states)
    for sess in sessions:
        n = len(sess)
        if n < 2:
            continue
        for k, u in enumerate(sess):
            if 0 <= u < num_states:
                s[u] += k / (n - 1)
                c[u] += 1
    return np.where(c > 0, s / np.maximum(c, 1), 0.5)


def _net_inflow_order(F, num_states):
    """Cheap data-only order: net inflow (source negative, sink positive)."""
    Fc = F.tocsr()
    inflow = np.asarray(Fc.sum(0)).ravel()
    outflow = np.asarray(Fc.sum(1)).ravel()
    return inflow - outflow


def extraction_ablation(funnel: SyntheticFunnel, lambda1: float = 1.0,
                        rho: float = 2.0, tau: float = 0.7,
                        top_k: int = 10) -> Dict:
    """Compare the Helmholtz--Hodge extraction against cheaper acyclic
    supports (dominance-thresholded topological sorts), under both
    regularizers. Isolates how much of the recovery depends on the Hodge
    projection versus on acyclicity and a good topological order alone.
    """
    train = _train_split(funnel)
    N = funnel.num_states
    pt = funnel.phi_true
    F = empirical_flow(train, N)
    Fd = dominance_filter(F, rho)
    b = divergence(F, N)
    sink_map = dict(funnel.sinks)

    _, retained, _ = hhd_dag_extract(F, Fd, N, tau=tau)
    dags = {
        "hhd": top_k_prune(retained, top_k),
        "topo_visit_position": top_k_prune(
            topological_dag(F, Fd, N, _mean_visit_position(train, N)), top_k),
        "topo_net_inflow": top_k_prune(
            topological_dag(F, Fd, N, _net_inflow_order(F, N)), top_k),
    }

    rows = []
    for name, dag in dags.items():
        m, frob = nilpotency_index(dag, N)
        # scored mask is per-extraction: the determined interior differs by DAG
        mask = scored_mask(funnel, train, dag)
        rec = {}
        for penalty in ("sobolev", "tikhonov"):
            phi, _ = solve_poisson(dag, N, sink_map, b,
                                   lambda1=lambda1, penalty=penalty)
            rec[penalty] = {
                "spearman": _rank_corr(pt[mask], phi[mask]),
                "pearson": float(pearsonr(pt[mask], phi[mask]).statistic)}
        rows.append({"extraction": name, "n_edges": len(dag),
                     "n_scored": int(mask.sum()),
                     "nilpotency_index": m, "acyclic": bool(frob == 0.0),
                     "sobolev": rec["sobolev"], "tikhonov": rec["tikhonov"],
                     "margin_spearman": rec["sobolev"]["spearman"]
                                        - rec["tikhonov"]["spearman"]})
    return {"rows": rows}


# ---------------------------------------------------------------------------
# E7: robustness of the regularizer contrast across seeds and configs
# ---------------------------------------------------------------------------
def regularizer_robustness(
        seeds=(42, 1, 7, 123, 2024),
        config_variants=(
            ("branches=3",  {"num_branches": 3}),
            ("branches=8",  {"num_branches": 8}),
            ("depth=30",    {"states_per_branch": 30}),
            ("depth=80",    {"states_per_branch": 80}),
            ("abandon=0.2", {"abandon_rate": 0.2}),
            ("abandon=0.5", {"abandon_rate": 0.5}),
            ("entropy=0.3", {"sink_entropy_param": 0.3}),
        ),
        base_seed: int = 42, lambda1: float = 1.0,
        num_sessions: int = 50_000,
        supports=("hhd", "topo_visit_position")) -> Dict:
    """Is the regularizer contrast a property of one fixed instrument, or does
    it hold across the generator's configuration space? For each of several
    variants (multiple seeds at the default config, plus single-axis
    perturbations of branch count, chain depth, abandonment, and sink entropy)
    we generate a funnel and score Sobolev vs Tikhonov (Spearman at ``lambda1``)
    under two extractions. The paper's claim is that the margin stays positive
    throughout; the *magnitude* of the recoverable correlation is regime-
    dependent (Sec. 7.3), but the regularizer's role is not.
    """
    variants = [(f"default (seed {s})", s, {}) for s in seeds]
    variants += [(label, base_seed, cfg) for (label, cfg) in config_variants]

    rows = []
    for label, seed, cfg in variants:
        funnel = generate_funnel(num_sessions=num_sessions, seed=seed, **cfg)
        by_ext = {r["extraction"]: r
                  for r in extraction_ablation(funnel, lambda1=lambda1)["rows"]}
        row = {"label": label, "seed": seed, "num_states": funnel.num_states,
               "sink_entropy_bits": float(funnel.sink_entropy)}
        for ext in supports:
            r = by_ext[ext]
            row[ext] = {"sobolev": r["sobolev"]["spearman"],
                        "tikhonov": r["tikhonov"]["spearman"],
                        "margin": r["margin_spearman"]}
        rows.append(row)

    summary = {}
    for ext in supports:
        sob = np.array([r[ext]["sobolev"] for r in rows])
        tik = np.array([r[ext]["tikhonov"] for r in rows])
        m = np.array([r[ext]["margin"] for r in rows])
        summary[ext] = {
            "sobolev_mean": float(sob.mean()), "sobolev_std": float(sob.std()),
            "tikhonov_mean": float(tik.mean()), "tikhonov_std": float(tik.std()),
            "margin_mean": float(m.mean()), "margin_std": float(m.std()),
            "margin_min": float(m.min()), "margin_max": float(m.max()),
            "n_positive": int((m > 0).sum()), "n_variants": int(len(m))}
    return {"lambda1": lambda1, "n_variants": len(rows),
            "rows": rows, "summary": summary}


def _layered_dag(n_layers: int, width: int, seed: int = 0):
    """A layered random DAG (width nodes per layer, edges layer i -> i+1) with a
    single pinned sink. Used only for the solver-scaling benchmark."""
    rng = np.random.default_rng(seed)
    edges = []
    N = n_layers * width + 1
    sink = N - 1
    for L in range(n_layers - 1):
        for a in range(width):
            u = L * width + a
            for _ in range(2):
                v = (L + 1) * width + int(rng.integers(width))
                edges.append((u, v, float(rng.uniform(0.5, 1.5))))
    for a in range(width):
        edges.append(((n_layers - 1) * width + a, sink, 1.0))
    b = np.zeros(N)
    b[:width] = -1.0
    b[sink] = float(width)
    return edges, N, {sink: 1.0}, b


def scaling_benchmark(sizes=((50, 20), (50, 40), (50, 80), (50, 160),
                             (100, 160), (200, 200)),
                      dense_max_interior: int = 4500, seed: int = 0) -> Dict:
    """Time the sparse-CG solve against dense Cholesky on layered DAGs of
    growing interior size, verifying agreement where both are feasible. Backs
    the scalability claim: the SPD system (\\cref{thm:spd}) lets CG scale to
    interiors where the dense factorization runs out of memory."""
    import time
    rows = []
    for (nl, w) in sizes:
        dag, N, sm, b = _layered_dag(nl, w, seed=seed)
        nI = N - 1
        t = time.time()
        phi_cg, _ = solve_poisson(dag, N, sm, b, 1.0, "sobolev",
                                  method="cg", cg_tol=1e-9)
        t_cg = time.time() - t
        t_chol = None
        max_diff = None
        if nI <= dense_max_interior:
            t = time.time()
            phi_ch, _ = solve_poisson(dag, N, sm, b, 1.0, "sobolev",
                                      method="cholesky")
            t_chol = time.time() - t
            max_diff = float(np.max(np.abs(phi_cg - phi_ch)))
        rows.append({"interior": nI, "edges": len(dag),
                     "cg_seconds": t_cg, "cholesky_seconds": t_chol,
                     "max_abs_diff": max_diff})
    return {"rows": rows}


def scaling_recovery(num_branches: int = 100, states_per_branch: int = 100,
                     shared_top_states: int = 100, num_sessions: int = 80_000,
                     lambda1: float = 1.0, seed: int = 42) -> Dict:
    """Recovery contrast at ~10^4 nodes via the sparse-CG solver (Sec 7.4).

    Generates a large planted funnel, extracts with the cheap topological sort,
    and scores graph-Sobolev vs Tikhonov on the determined interior, timing the
    CG solve. The gauge-stable CG path converges to the same minimum-norm
    estimate as the dense solve, so the regularizer contrast is preserved at
    scale."""
    import time
    f = generate_funnel(num_branches=num_branches,
                        states_per_branch=states_per_branch,
                        shared_top_states=shared_top_states,
                        num_sessions=num_sessions, seed=seed)
    train = _train_split(f)
    N = f.num_states
    F = empirical_flow(train, N)
    Fd = dominance_filter(F, 2.0)
    b = divergence(F, N)
    dag = top_k_prune(topological_dag(F, Fd, N, _mean_visit_position(train, N)), 10)
    sink_map = dict(f.sinks)
    pt = f.phi_true
    mask = scored_mask(f, train, dag)
    out = {"num_states": N, "num_dag_edges": len(dag), "n_scored": int(mask.sum())}
    for penalty in ("sobolev", "tikhonov"):
        t = time.time()
        phi, _ = solve_poisson(dag, N, sink_map, b, lambda1=lambda1,
                               penalty=penalty, method="cg", cg_tol=1e-9)
        out[penalty] = {"spearman": _rank_corr(pt[mask], phi[mask]),
                        "cg_seconds": time.time() - t}
    return out
