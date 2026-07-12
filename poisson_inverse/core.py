"""
Discrete Poisson inverse problem on a directed acyclic graph.

Pipeline (all on the macroscopic state space):

  1. empirical_flow         F[u,v] = # of u->v transitions
  2. dominance_filter       keep (u,v) iff f_uv >= rho * f_vu
  3. hhd_dag_extract         gradient component of the discrete
                             Helmholtz--Hodge decomposition -> acyclic DAG
  4. divergence              b[i] = inflow(i) - outflow(i)
  5. solve_poisson           (L^T L + lambda1 * R) phi = L^T b, with
                             Dirichlet boundaries; R is the graph-Sobolev
                             (Dirichlet-energy) or the Tikhonov penalty
  6. poisson_residual        |L phi - b|, used for blind sink discovery

The two regularizers share one solver and differ only in the penalty
matrix R, which is the whole point of the paper.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from .synthetic import collapse_consecutive

Edge = Tuple[int, int, float]


# ---------------------------------------------------------------------------
# Stage 1-2: empirical flow and dominance filter
# ---------------------------------------------------------------------------
def empirical_flow(sessions: List[List[int]], num_states: int) -> sp.csc_matrix:
    """F[u,v] = count of u->v transitions over collapsed sessions."""
    rows, cols, vals = [], [], []
    for s in sessions:
        c = collapse_consecutive(s)
        for i in range(len(c) - 1):
            u, v = c[i], c[i + 1]
            if u != v and 0 <= u < num_states and 0 <= v < num_states:
                rows.append(u); cols.append(v); vals.append(1.0)
    F = sp.coo_matrix((vals, (rows, cols)), shape=(num_states, num_states)).tocsc()
    F.sum_duplicates()
    return F


def dominance_filter(F: sp.csc_matrix, rho: float) -> sp.csc_matrix:
    """Retain (u,v) iff f_uv >= rho * f_vu (Definition: directional dominance)."""
    F_csr = F.tocsr()
    F_coo = F.tocoo()
    r, c, d = [], [], []
    for u, v, w in zip(F_coo.row, F_coo.col, F_coo.data):
        if w >= rho * F_csr[v, u]:
            r.append(u); c.append(v); d.append(w)
    return sp.coo_matrix((d, (r, c)), shape=F.shape).tocsc()


# ---------------------------------------------------------------------------
# Stage 3: HHD gradient projection -> DAG
# ---------------------------------------------------------------------------
def _cg_normal(G: sp.csc_matrix, Gt: sp.csc_matrix, b: np.ndarray, n: int,
               tol: float = 1e-8, max_iter: int = 200) -> np.ndarray:
    """Conjugate gradient on the normal equations G^T G x = G^T b."""
    rhs = np.asarray(Gt @ b).ravel()
    x = np.zeros(n)
    r = rhs.copy()
    p = r.copy()
    rs_old = float(r @ r)
    thresh = tol * max(float(np.linalg.norm(rhs)), 1.0)
    for _ in range(max_iter):
        if np.sqrt(rs_old) <= thresh:
            break
        Gp = np.asarray(G @ p).ravel()
        Ap = np.asarray(Gt @ Gp).ravel()
        pAp = float(p @ Ap)
        if abs(pAp) < 1e-30:
            return x
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(r @ r)
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


def hhd_dag_extract(F: sp.csc_matrix, F_dom: sp.csc_matrix, num_states: int,
                    tau: float = 0.7, epsilon: float = 1e-9):
    """Gradient component of the discrete HHD on the net-flow support.

    Solves G^T G phi0 = G^T omega where omega_uv = f_uv - f_vu is the
    skew-symmetric edge flow and G is the signed incidence operator.
    Retains an edge in the DAG iff its recovered gradient agrees in sign
    ((G phi0)_uv > 0) and dominates the raw flow (delta >= tau).

    Returns
    -------
    phi0      : (num_states,) unconstrained gradient potential
    retained  : list of (u, v, w) DAG edges
    edge_grad : list of (u, v, omega_uv, grad_uv) over the support, for the
                gradient-direction-preservation diagnostic
    """
    F_csr = F.tocsr()
    D_coo = F_dom.tocoo()
    edges = []  # (u, v, omega, w)
    for u, v, w in zip(D_coo.row, D_coo.col, D_coo.data):
        f_uv = F_csr[u, v]
        f_vu = F_csr[v, u]
        if f_uv > f_vu:
            edges.append((int(u), int(v), float(f_uv - f_vu), float(w)))

    if not edges:
        return np.zeros(num_states), [], []

    n_e = len(edges)
    g_r, g_c, g_d = [], [], []
    omega = np.zeros(n_e)
    for e, (u, v, om, _) in enumerate(edges):
        g_r += [e, e]; g_c += [v, u]; g_d += [1.0, -1.0]
        omega[e] = om
    G = sp.csc_matrix((g_d, (g_r, g_c)), shape=(n_e, num_states))
    Gt = G.T.tocsc()
    phi0 = _cg_normal(G, Gt, omega, num_states)

    retained, edge_grad = [], []
    for (u, v, om, w) in edges:
        grad = phi0[v] - phi0[u]
        delta = abs(grad) / (abs(om) + epsilon)
        edge_grad.append((u, v, om, grad))
        if grad > 0.0 and delta >= tau:
            retained.append((u, v, w))
    return phi0, retained, edge_grad


def topological_dag(F: sp.csc_matrix, F_dom: sp.csc_matrix, num_states: int,
                    order: np.ndarray) -> List[Edge]:
    """Cheap acyclic support: keep net-forward dominant edges that agree with
    a supplied node ``order`` (a data-derived ``downstreamness'' score).

    This is the extraction baseline used in the ablation of
    Section~\\ref{sec:exp-ablation}: it requires no Helmholtz--Hodge solve,
    only the dominance filter and a topological order. Acyclicity is
    guaranteed because every retained edge increases ``order``.
    """
    F_csr = F.tocsr()
    D = F_dom.tocoo()
    edges: List[Edge] = []
    for u, v, w in zip(D.row, D.col, D.data):
        if F_csr[u, v] > F_csr[v, u] and order[u] < order[v]:
            edges.append((int(u), int(v), float(w)))
    return edges


def top_k_prune(edges: List[Edge], top_k: int) -> List[Edge]:
    """Keep the top-k highest-weight outgoing edges per source."""
    if not edges:
        return edges
    per_src: Dict[int, List[Tuple[int, float]]] = {}
    for (u, v, w) in edges:
        per_src.setdefault(u, []).append((v, w))
    out = []
    for u, lst in per_src.items():
        lst.sort(key=lambda x: -x[1])
        out.extend((u, v, w) for (v, w) in lst[:top_k])
    return out


# ---------------------------------------------------------------------------
# Stage 4: divergence
# ---------------------------------------------------------------------------
def divergence(F: sp.csc_matrix, num_states: int) -> np.ndarray:
    """b[i] = inflow(i) - outflow(i), from the raw flow F."""
    b = np.zeros(num_states)
    F_coo = F.tocoo()
    for r, c, v in zip(F_coo.row, F_coo.col, F_coo.data):
        b[c] += v
        b[r] -= v
    return b


# ---------------------------------------------------------------------------
# Stage 5: regularized Poisson solve (Dirichlet-anchored)
# ---------------------------------------------------------------------------
def _directed_laplacian(dag_edges: List[Edge], num_states: int) -> np.ndarray:
    L = np.zeros((num_states, num_states))
    for (u, v, w) in dag_edges:
        L[u, u] += w
        L[u, v] -= w
    return L


def _graph_laplacian(dag_edges: List[Edge], num_states: int) -> np.ndarray:
    """Symmetric, conductance-weighted graph Laplacian L_G.

    phi^T L_G phi = sum_{(u,v) in E} w_uv (phi_v - phi_u)^2, the discrete
    Dirichlet energy. (Weighted form -- matches the implementation used
    for every number in the paper.)
    """
    LG = np.zeros((num_states, num_states))
    for (u, v, w) in dag_edges:
        LG[u, u] += w
        LG[v, v] += w
        LG[u, v] -= w
        LG[v, u] -= w
    return LG


def _directed_laplacian_sparse(dag_edges: List[Edge], num_states: int) -> sp.csr_matrix:
    """Sparse directed Laplacian, for the large-scale conjugate-gradient path."""
    rows, cols, vals = [], [], []
    for (u, v, w) in dag_edges:
        rows += [u, u]; cols += [u, v]; vals += [w, -w]
    return sp.csr_matrix((vals, (rows, cols)), shape=(num_states, num_states))


def _graph_laplacian_sparse(dag_edges: List[Edge], num_states: int) -> sp.csr_matrix:
    """Sparse conductance-weighted graph Laplacian L_G (Dirichlet energy)."""
    rows, cols, vals = [], [], []
    for (u, v, w) in dag_edges:
        rows += [u, v, u, v]
        cols += [u, v, v, u]
        vals += [w, w, -w, -w]
    return sp.csr_matrix((vals, (rows, cols)), shape=(num_states, num_states))


def _cg_solve_spd(A: sp.csr_matrix, rhs: np.ndarray,
                  tol: float = 1e-8, maxiter: int = None):
    """Conjugate-gradient solve of an SPD sparse system (Theorem 1 guarantees
    SPD for lambda1 > 0). Robust to the scipy tol/rtol API change."""
    import scipy.sparse.linalg as sla
    try:
        x, info = sla.cg(A, rhs, rtol=tol, atol=0.0, maxiter=maxiter)
    except TypeError:                       # older scipy: tol= instead of rtol=
        x, info = sla.cg(A, rhs, tol=tol, maxiter=maxiter)
    return x, info


def solve_poisson(dag_edges: List[Edge], num_states: int,
                  sink_map: Dict[int, float], b: np.ndarray,
                  lambda1: float = 0.0, penalty: str = "sobolev",
                  method: str = "auto", cg_tol: float = 1e-8,
                  cg_maxiter: int = None, rtol: float = 1e-9
                  ) -> Tuple[np.ndarray, float]:
    """Solve the Dirichlet-anchored regularized Poisson problem.

        min_phi  ||L phi - b||^2 / ||b||^2  +  lambda1 * R(phi)

    with phi pinned to ``sink_map`` on the boundary. ``penalty`` selects
    the regularizer R:

      "sobolev"  : phi^T L_G phi   (gauge-invariant Dirichlet energy)
      "tikhonov" : ||phi||^2       (gauge-dependent ridge)
      "none"     : lambda1 ignored

    ``method`` selects the linear solver:

      "auto"     : dense symmetric eigensolve for small interiors, sparse CG
                   for large
      "cholesky" : dense gauge-stable eigensolve (exact; used for every table
                   in the paper -- see below)
      "cg"       : sparse conjugate gradients (Theorem 1 guarantees SPD, so
                   CG applies; O(kappa(M) * nnz) per iteration, no dense fill)

    The dense path is a *gauge-stable* solve: interior nodes with no directed
    path to the boundary make the reduced operator rank-deficient (Theorem 1's
    reachability hypothesis fails for them), so we invert only its numerically
    nonzero spectrum (relative tolerance ``rtol``). This is the minimum-norm
    solution; the gauge/constant mode lies in the range and is reproduced
    exactly, whereas a Cholesky solve with a small diagonal floor amplifies
    roundoff along the near-null directions and breaks origin-invariance.

    Returns (phi, flow_residual).
    """
    if penalty not in ("sobolev", "tikhonov", "none"):
        raise ValueError(f"unknown penalty {penalty!r}")
    if method not in ("auto", "cholesky", "cg"):
        raise ValueError(f"unknown method {method!r}")
    if not dag_edges:
        phi = np.zeros(num_states)
        for k, v in sink_map.items():
            phi[k] = v
        return phi, 0.0

    is_b = np.zeros(num_states, dtype=bool)
    phi_b_full = np.zeros(num_states)
    for k, v in sink_map.items():
        if 0 <= k < num_states:
            is_b[k] = True
            phi_b_full[k] = v
    interior = np.where(~is_b)[0]
    boundary = np.where(is_b)[0]
    nI = len(interior)
    if nI == 0:
        return phi_b_full.copy(), 0.0

    if method == "auto":
        method = "cg" if nI > 1500 else "cholesky"

    # ---- sparse conjugate-gradient path (large interiors) ----------------
    if method == "cg":
        L = _directed_laplacian_sparse(dag_edges, num_states).tocsr()
        LII = L[interior][:, interior]
        LIB = L[interior][:, boundary]
        bI = b[interior]
        phiB = phi_b_full[boundary]
        b_eff = bI - np.asarray(LIB @ phiB).ravel()
        b_scale = max(float(b_eff @ b_eff), 1e-12)
        A = (LII.T @ LII) / b_scale
        rhs = np.asarray(LII.T @ b_eff).ravel() / b_scale
        if lambda1 > 0 and penalty == "sobolev":
            LG = _graph_laplacian_sparse(dag_edges, num_states).tocsr()
            LG_II = LG[interior][:, interior]
            LG_IB = LG[interior][:, boundary]
            scale = max(float(LG_II.diagonal().sum()), 1e-12)
            A = A + (lambda1 / scale) * LG_II
            rhs = rhs - (lambda1 / scale) * np.asarray(LG_IB @ phiB).ravel()
        elif lambda1 > 0 and penalty == "tikhonov":
            A = A + lambda1 * sp.eye(nI, format="csr")
        # Gauge-stable: no diagonal floor. On the consistent PSD system CG from
        # x0=0 stays in the range and converges to the min-norm solution,
        # matching the dense eigen-truncated path (a floor would instead pin the
        # near-null gauge/no-path directions to roundoff, breaking invariance).
        A = A.tocsr()
        phi_I, _ = _cg_solve_spd(A, rhs, tol=cg_tol, maxiter=cg_maxiter)
        phi = np.zeros(num_states)
        phi[interior] = phi_I
        phi[boundary] = phiB
        residual = float(np.linalg.norm(
            np.asarray(LII @ phi_I).ravel()
            + np.asarray(LIB @ phiB).ravel() - bI))
        return phi, residual

    # ---- dense Cholesky path (small interiors; exact) -------------------
    L = _directed_laplacian(dag_edges, num_states)
    LII = L[np.ix_(interior, interior)]
    LIB = L[np.ix_(interior, boundary)]
    bI = b[interior]
    phiB = phi_b_full[boundary]
    b_eff = bI - LIB @ phiB

    b_scale = max(float(b_eff @ b_eff), 1e-12)
    A = (LII.T @ LII) / b_scale
    rhs = (LII.T @ b_eff) / b_scale

    if lambda1 > 0 and penalty == "sobolev":
        LG = _graph_laplacian(dag_edges, num_states)
        LG_II = LG[np.ix_(interior, interior)]
        LG_IB = LG[np.ix_(interior, boundary)]
        scale = max(float(np.trace(LG_II)), 1e-12)   # trace-normalize
        A = A + (lambda1 / scale) * LG_II
        rhs = rhs - (lambda1 / scale) * (LG_IB @ phiB)
    elif lambda1 > 0 and penalty == "tikhonov":
        A = A + lambda1 * np.eye(nI)                  # gauge-dependent ridge

    # Gauge-stable solve: eigendecompose the SPD-in-theory operator and invert
    # only the numerically nonzero spectrum. No-path interior nodes make A
    # rank-deficient; truncating the null space yields the min-norm solution and
    # keeps the gauge mode (which is in the range) exactly, instead of a
    # diagonal floor that amplifies roundoff along the near-null directions.
    w, V = np.linalg.eigh(A)
    keep = w > rtol * w.max()
    coef = V.T @ rhs
    coef[keep] /= w[keep]
    coef[~keep] = 0.0
    phi_I = V @ coef

    phi = np.zeros(num_states)
    phi[interior] = phi_I
    phi[boundary] = phiB
    residual = float(np.linalg.norm(LII @ phi_I + LIB @ phiB - bI))
    return phi, residual


# ---------------------------------------------------------------------------
# Reachability to the boundary (Theorem 1's hypothesis)
# ---------------------------------------------------------------------------
def reachable_to_boundary(dag_edges: List[Edge], num_states: int,
                          boundary: set) -> np.ndarray:
    """Boolean mask of nodes with a directed path to some boundary node.

    Theorem 1 requires every interior node to reach the boundary; nodes that
    do not are undetermined by the Dirichlet data (the reduced operator is
    singular on them). Used both by the gauge-stable solve's interpretation and
    to restrict scoring to the *determined* interior.
    """
    from collections import defaultdict
    adj = defaultdict(list)
    for (u, v, _) in dag_edges:
        adj[u].append(v)
    boundary = set(boundary)
    mask = np.zeros(num_states, dtype=bool)
    for start in range(num_states):
        seen = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            if x in boundary:
                mask[start] = True
                break
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
    return mask


# ---------------------------------------------------------------------------
# Stage 6: blind sink discovery via the Poisson residual
# ---------------------------------------------------------------------------
def poisson_residual(dag_edges: List[Edge], num_states: int,
                     phi: np.ndarray, b: np.ndarray) -> np.ndarray:
    """r[u] = |[L phi]_u - b_u|. Concentrates at no-out-edge sinks."""
    L = _directed_laplacian(dag_edges, num_states)
    return np.abs(L @ phi - b)


# ---------------------------------------------------------------------------
# Structural diagnostic: nilpotency of the sub-stochastic operator
# ---------------------------------------------------------------------------
def substochastic_operator(dag_edges: List[Edge], num_states: int) -> np.ndarray:
    """Row-stochastic transition operator A on the DAG (sinks absorbing)."""
    A = np.zeros((num_states, num_states))
    out = np.zeros(num_states)
    for (u, v, w) in dag_edges:
        A[u, v] += w
        out[u] += w
    for i in range(num_states):
        if out[i] > 0:
            A[i] /= out[i]
    return A


def nilpotency_index(dag_edges: List[Edge], num_states: int,
                     tol: float = 1e-12) -> Tuple[int, float]:
    """Smallest m with A^m = 0, plus ||A^m||_F at that m.

    Returns (m, frobenius_norm). On an acyclic graph A is nilpotent and
    m is bounded by the longest path length.
    """
    A = substochastic_operator(dag_edges, num_states)
    M = A.copy()
    for k in range(1, num_states + 2):
        nrm = float(np.linalg.norm(M, "fro"))
        if nrm <= tol:
            return k, nrm
        M = M @ A
    return num_states + 2, float(np.linalg.norm(M, "fro"))
