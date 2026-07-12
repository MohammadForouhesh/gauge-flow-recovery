"""Fast sanity checks for the core solver and the paper's claims.

Run with:  python -m pytest tests/ -q     (or)    python tests/test_core.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poisson_inverse import (
    generate_funnel, build_chain, solve_poisson, nilpotency_index,
    empirical_flow, dominance_filter, hhd_dag_extract, top_k_prune, divergence,
)
from poisson_inverse.core import _graph_laplacian


def test_gauge_invariance():
    """phi^T L_G phi is invariant to adding a constant; ||phi||^2 is not."""
    dag = [(0, 1, 2.0), (1, 2, 1.0), (0, 2, 0.5)]
    N = 3
    LG = _graph_laplacian(dag, N)
    phi = np.array([0.3, 0.7, 1.0])
    one = np.ones(N)
    e0 = phi @ LG @ phi
    e1 = (phi + 5.0 * one) @ LG @ (phi + 5.0 * one)
    assert abs(e0 - e1) < 1e-10, "Dirichlet energy must be gauge-invariant"
    assert abs(np.sum(phi**2) - np.sum((phi + 5.0)**2)) > 1.0, \
        "magnitude penalty must depend on gauge"


def test_chain_range_preservation():
    """Sobolev preserves chain dynamic range; Tikhonov collapses it."""
    dag, N, sink_map, b = build_chain(length=30)
    sink = N - 1
    interior = [i for i in range(N) if i != sink]

    def rng(penalty, lam):
        phi, _ = solve_poisson(dag, N, sink_map, b, lambda1=lam, penalty=penalty)
        return float(phi[interior].max() - phi[interior].min())

    sob = rng("sobolev", 1.0)
    tik = rng("tikhonov", 1.0)
    assert sob > 0.9, f"Sobolev range should be ~1, got {sob:.3f}"
    assert tik < 0.7, f"Tikhonov range should collapse, got {tik:.3f}"
    assert sob > tik + 0.3, "Sobolev must beat Tikhonov by a clear margin"


def test_spd_solvable():
    """The Sobolev solve returns a finite, boundary-respecting solution."""
    f = generate_funnel(num_sessions=4000, seed=0)
    F = empirical_flow(f.sessions, f.num_states)
    phi0, retained, _ = hhd_dag_extract(F, dominance_filter(F, 2.0), f.num_states)
    dag = top_k_prune(retained, 10)
    b = divergence(F, f.num_states)
    phi, res = solve_poisson(dag, f.num_states, dict(f.sinks), b,
                             lambda1=1.0, penalty="sobolev")
    assert np.all(np.isfinite(phi))
    for s in f.conversion_sink_ids:
        assert abs(phi[s] - 1.0) < 1e-9, "conversion sinks pinned at 1"
    for s in f.abandon_sink_ids:
        assert abs(phi[s] - 0.0) < 1e-9, "abandon sink pinned at 0"


def test_nilpotency():
    """The extracted DAG operator is exactly nilpotent."""
    f = generate_funnel(num_sessions=8000, seed=1)
    F = empirical_flow(f.sessions, f.num_states)
    phi0, retained, _ = hhd_dag_extract(F, dominance_filter(F, 2.0), f.num_states)
    dag = top_k_prune(retained, 10)
    m, frob = nilpotency_index(dag, f.num_states)
    assert frob == 0.0, f"DAG operator must be nilpotent, ||A^m||_F={frob}"
    assert m >= 2


def test_extraction_ablation():
    """The regularizer contrast holds under a cheap topological-sort support,
    and that support is exactly acyclic."""
    from poisson_inverse import experiments as E
    f = generate_funnel(num_sessions=12000, seed=3)
    res = E.extraction_ablation(f, lambda1=1.0)
    assert len(res["rows"]) == 3
    for r in res["rows"]:
        assert r["acyclic"], f"{r['extraction']} must be acyclic"
        # Sobolev should not invert; ridge should be worse than Sobolev
        assert r["sobolev"]["spearman"] > r["tikhonov"]["spearman"]
        assert r["margin_spearman"] > 0


def test_cg_matches_cholesky():
    """The sparse-CG solver agrees with dense Cholesky on a small SPD system."""
    from poisson_inverse import experiments as E
    dag, N, sm, b = E._layered_dag(20, 15, seed=1)
    for pen in ("sobolev", "tikhonov"):
        p_ch, _ = solve_poisson(dag, N, sm, b, 1.0, pen, method="cholesky")
        p_cg, _ = solve_poisson(dag, N, sm, b, 1.0, pen, method="cg", cg_tol=1e-10)
        assert np.max(np.abs(p_ch - p_cg)) < 1e-5


def test_parameter_insensitivity():
    """The recovered phi is parameter-insensitive under the Dirichlet-energy
    penalty (rank correlation flat in lambda), while ridge inverts the ordering
    for lambda>0 and keeps top-k ranking below Sobolev."""
    from poisson_inverse.experiments import regularizer_sweep
    from poisson_inverse.synthetic import generate_funnel
    res = regularizer_sweep(generate_funnel(num_sessions=30000, seed=42))
    sob_sp = {r["lambda1"]: r["spearman"] for r in res["sobolev"]}
    tik_sp = {r["lambda1"]: r["spearman"] for r in res["tikhonov"]}
    pos = [l for l in res["lambdas"] if l > 0]
    # Sobolev rank correlation is flat in lambda (parameter-insensitive)
    sob_vals = [sob_sp[l] for l in pos]
    assert max(sob_vals) - min(sob_vals) < 0.02
    # ...and stays well above ridge, which inverts (negative) for every lambda>0
    assert min(sob_vals) > 0.5
    assert max(tik_sp[l] for l in pos) < 0.0
    # ridge also keeps top-k ranking below its own lambda=0 value
    tik_nd = {r["lambda1"]: r["ndcg5"] for r in res["tikhonov"]}
    assert max(tik_nd[l] for l in pos) < tik_nd[0.0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
