#!/usr/bin/env python3
"""Directed-GNN oversmoothing baseline for the OTTO conversion task (Sec. 8.4).

Validates the oversmoothing analogy of Sec. 2: a directed GCN on the OTTO graph
loses node distinction with depth, exactly as ridge loses it with lambda1.

  (1) pure propagation:  Dirichlet energy of Asl^k X collapses with depth k
  (2) trained GNN:       held-out conversion ROC-AUC degrades with depth, and
                         node-embedding Dirichlet energy collapses.

The GNN pools node embeddings over the interior (non-sink) states a session
visits and predicts conversion -- the same leakage-guarded target and pooling as
the potential features, so the comparison is apples-to-apples. Requires PyTorch
(optional dependency); everything else uses the released pipeline.

Usage:
    python scripts/run_gnn.py --train /path/to/otto/train.jsonl [--max_sessions 120000]
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from poisson_inverse.loaders import load_otto
from poisson_inverse.core import (empirical_flow, dominance_filter, top_k_prune,
                                  topological_dag)
from poisson_inverse.experiments import _mean_visit_position


def _norm_rows(M):
    d = np.asarray(M.sum(1)).ravel(); d[d == 0] = 1.0
    return (sp.diags(1.0 / d) @ M).tocsr()


def build_graph(ds, train_sessions, rho=2.0, top_k=10):
    N = ds.num_states
    F = empirical_flow(train_sessions, N)
    Fd = dominance_filter(F, rho)
    dag = top_k_prune(topological_dag(F, Fd, N, _mean_visit_position(train_sessions, N)), top_k)
    A = sp.lil_matrix((N, N))
    for (u, v, w) in dag:
        A[u, v] = w
    A = A.tocsr()
    As = ((A + A.T) > 0).astype(np.float64)
    d = np.asarray(As.sum(1)).ravel(); d[d == 0] = 1.0
    Dm12 = sp.diags(1.0 / np.sqrt(d))
    Lsym = sp.csr_matrix(sp.eye(N) - Dm12 @ As @ Dm12)          # oversmoothing metric
    return A, _norm_rows(A), _norm_rows(A.T.tocsr()), Lsym, len(dag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="OTTO train.jsonl")
    ap.add_argument("--max_sessions", type=int, default=120_000)
    ap.add_argument("--num_categories", type=int, default=50)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as Fn
    except ImportError:
        sys.exit("PyTorch is required for the GNN baseline: pip install torch")
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ds = load_otto(args.train, num_categories=args.num_categories,
                   max_sessions=args.max_sessions, seed=42, verbose=False)
    N = ds.num_states
    conv = set(ds.conversion_sink_ids)
    sinks = conv | set(ds.abandon_sink_ids) | {k for k, _ in ds.sinks}
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(ds.sessions)); cut = int(0.7 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    A, Aout, Ain, Lsym, n_edges = build_graph(ds, [ds.sessions[i] for i in tr])

    def to_t(M):
        M = M.tocoo()
        return torch.sparse_coo_tensor(np.vstack([M.row, M.col]),
                                       M.data.astype(np.float32), M.shape).coalesce()

    def pool(sess_list):
        rows, cols = [], []
        for i, s in enumerate(sess_list):
            for u in set(u for u in s if u not in sinks):
                rows.append(i); cols.append(u)
        P = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(sess_list), N))
        return to_t(_norm_rows(P))

    def labels(sess_list):
        return torch.tensor([1.0 if (s and s[-1] in conv) else 0.0 for s in sess_list])

    Aout_t, Ain_t, Lsym_t = to_t(Aout), to_t(Ain), to_t(Lsym)
    Ptr, Pte = pool([ds.sessions[i] for i in tr]), pool([ds.sessions[i] for i in te])
    ytr, yte = labels([ds.sessions[i] for i in tr]), labels([ds.sessions[i] for i in te]).numpy()

    # constant/gauge mode: smallest eigenvector of L_sym is d^{1/2} (unit-normalized)
    As = ((A + A.T) > 0).astype(np.float64)
    deg = np.asarray(As.sum(1)).ravel(); deg[deg == 0] = 1.0
    u0 = torch.tensor(np.sqrt(deg), dtype=torch.float32); u0 = u0 / u0.norm()

    def dir_energy(H):
        H = H.detach()
        return float(torch.trace(H.t() @ torch.sparse.mm(Lsym_t, H)) / (H.pow(2).sum() + 1e-12))

    print(f"OTTO graph: N={N} edges={n_edges} train={len(tr)} test={len(te)} "
          f"conv={yte.mean():.3f}")

    # (1) pure propagation collapse (self-looped forward operator)
    Aself = sp.csr_matrix(Aout + sp.eye(N))
    d = np.asarray(Aself.sum(1)).ravel(); d[d == 0] = 1.0
    Aself = to_t(sp.diags(1.0 / d) @ Aself)
    H = torch.randn(N, args.dim)
    print("\n[pure propagation] depth k : Dirichlet energy of Asl^k X")
    for k in range(0, max(args.depths) + 1):
        if k in (0, 1, 2, 4, 8, 16):
            print(f"   k={k:2d}  {dir_energy(H):.4f}")
        H = torch.sparse.mm(Aself, H)

    # (2) trained directed GCN, AUC + energy vs depth. ``mode`` selects the
    # anti-oversmoothing intervention: "gauge" removes the constant/gauge mode
    # each layer (the per-layer analogue of L_G 1 = 0), "gauge_scale" also
    # rescales (PairNorm-style). This is the gauge-invariance principle applied
    # to the network itself.
    class DiGCN(nn.Module):
        def __init__(self, depth, dim, mode):
            super().__init__(); self.emb = nn.Embedding(N, dim); self.mode = mode
            self.Wo = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
            self.Wi = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(depth)])
            self.out = nn.Linear(dim, 1)
        def embed(self):
            H = self.emb.weight
            for l in range(len(self.Wo)):
                H = Fn.relu(self.Wo[l](torch.sparse.mm(Aout_t, H))
                            + self.Wi[l](torch.sparse.mm(Ain_t, H)))
                if self.mode in ("gauge", "gauge_scale"):
                    H = H - torch.outer(u0, (u0 @ H))               # remove gauge mode
                if self.mode == "gauge_scale":
                    H = H * (N ** 0.5) / (H.norm() + 1e-6)
            return H
        def forward(self, P):
            H = self.embed(); return self.out(torch.sparse.mm(P, H)).squeeze(-1), H

    for mode in ("vanilla", "gauge", "gauge_scale"):
        print(f"\n[trained directed GNN: {mode}]  depth : test_AUC  node_energy")
        for depth in args.depths:
            torch.manual_seed(args.seed)
            m = DiGCN(depth, args.dim, mode)
            opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=1e-4)
            for _ in range(args.epochs):
                m.train(); opt.zero_grad()
                lo, _ = m(Ptr)
                Fn.binary_cross_entropy_with_logits(lo, ytr).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                lo, Hemb = m(Pte)
                auc = roc_auc_score(yte, torch.sigmoid(lo).numpy())
            print(f"   L={depth:2d}   {auc:.4f}   {dir_energy(Hemb):.4f}", flush=True)
    print("\n(reference potential features: Sobolev 0.827, ridge 0.752, raw 0.828)")


if __name__ == "__main__":
    main()
