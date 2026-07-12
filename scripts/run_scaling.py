#!/usr/bin/env python3
"""Solver-scaling benchmark: sparse conjugate gradients vs dense Cholesky on
layered DAGs of growing interior size. Reproduces the scalability table.

The SPD guarantee (Theorem 1) lets CG scale to interiors where the dense
factorization runs out of memory. Run:

    python scripts/run_scaling.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poisson_inverse import experiments as E


def main():
    res = E.scaling_benchmark()
    print(f"{'interior':>9} {'edges':>8} {'CG (s)':>8} {'eigensolve (s)':>15} "
          f"{'max|diff|':>10}")
    for r in res["rows"]:
        chol = "OOM/skip" if r["cholesky_seconds"] is None \
            else f"{r['cholesky_seconds']:.2f}"
        md = "--" if r["max_abs_diff"] is None else f"{r['max_abs_diff']:.1e}"
        print(f"{r['interior']:>9} {r['edges']:>8} {r['cg_seconds']:>8.3f} "
              f"{chol:>15} {md:>10}")

    print("\n[*] recovery contrast at ~10^4 nodes (Sec 7.4) ...")
    rec = E.scaling_recovery()
    print(f"    |V|={rec['num_states']}, n_scored={rec['n_scored']}: "
          f"Sobolev Spearman={rec['sobolev']['spearman']:+.3f} "
          f"(CG {rec['sobolev']['cg_seconds']:.3f}s), "
          f"Tikhonov Spearman={rec['tikhonov']['spearman']:+.3f}")

    with open("scaling_results.json", "w") as fh:
        json.dump({"benchmark": res, "recovery": rec}, fh, indent=2)
    print("\n[*] wrote scaling_results.json")


if __name__ == "__main__":
    sys.exit(main())
