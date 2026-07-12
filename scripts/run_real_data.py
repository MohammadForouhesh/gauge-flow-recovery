#!/usr/bin/env python3
"""Run the real-data experiments on RetailRocket, Trivago, or OTTO.

Examples
--------
    python scripts/run_real_data.py retailrocket \
        --events /data/retailrocket/events.csv

    python scripts/run_real_data.py trivago \
        --train /data/trivago/train.csv --fraction 0.5

    python scripts/run_real_data.py otto \
        --train /data/otto/train.jsonl --num_categories 50 \
        --max_sessions 200000

Writes results_<dataset>.json and prints a readable summary.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poisson_inverse.loaders import (
    load_retailrocket, load_trivago, load_otto)
from poisson_inverse import real_experiments as R


def build_dataset(args):
    if args.dataset == "retailrocket":
        return load_retailrocket(args.events, fraction=args.fraction,
                                 seed=args.seed)
    if args.dataset == "trivago":
        return load_trivago(args.train, fraction=args.fraction, seed=args.seed)
    if args.dataset == "otto":
        return load_otto(args.train, fraction=args.fraction,
                         num_categories=args.num_categories,
                         cart_sink_phi=args.cart_sink_phi,
                         min_item_interactions=args.min_item_interactions,
                         max_sessions=args.max_sessions, seed=args.seed)
    raise ValueError(args.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["retailrocket", "trivago", "otto"])
    ap.add_argument("--events", help="RetailRocket events.csv")
    ap.add_argument("--train", help="Trivago train.csv or OTTO train.jsonl")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--num_categories", type=int, default=50,
                    help="OTTO co-visit clusters (0 = minimal 5-state graph)")
    ap.add_argument("--cart_sink_phi", type=float, default=0.6)
    ap.add_argument("--min_item_interactions", type=int, default=5,
                    help="OTTO: drop items seen fewer than this many times")
    ap.add_argument("--max_sessions", type=int, default=0,
                    help="OTTO: cap sessions for laptop-scale runs")
    ap.add_argument("--rho", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--lambda1", type=float, default=1.0)
    ap.add_argument("--n_boot", type=int, default=20)
    ap.add_argument("--no_downstream", action="store_true",
                    help="skip R5 (session-level conversion prediction)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = build_dataset(args)
    kw = dict(rho=args.rho, tau=args.tau, top_k=args.top_k)

    print(f"\n[R1] potential recovery ({args.dataset}) ...")
    r1 = R.recover_potential(ds, lambda1=args.lambda1, penalty="sobolev", **kw)
    print("[R2] regularizer contrast ...")
    r2 = R.regularizer_contrast(ds, lambda1=args.lambda1, **kw)
    print("[R3] sink discovery (known boundary) ...")
    r3 = R.sink_discovery_real(ds, lambda1=args.lambda1, **kw)
    print(f"[R4] bootstrap stability (n={args.n_boot}) ...")
    r4 = R.bootstrap_stability(ds, n_boot=args.n_boot, lambda1=args.lambda1,
                               seed=args.seed, **kw)
    r5 = None
    if not args.no_downstream:
        print("[R5] downstream conversion prediction (Sobolev vs ridge) ...")
        r5 = R.conversion_prediction(ds, lambda1=args.lambda1, **kw)

    results = {"dataset": ds.name, "summary": ds.summary(),
               "R1_potential": r1, "R2_regularizer_contrast": r2,
               "R3_sink_discovery": r3, "R4_bootstrap": r4,
               "R5_downstream": r5}
    out = args.out or f"results_{args.dataset}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Console summary
    print("\n==================== SUMMARY ====================")
    print(ds.summary())
    print(f"\nR1  recovered phi (top of {r1['num_dag_edges']}-edge DAG):")
    for row in r1["phi_table"][:8]:
        print(f"      {row['phi']:+.3f}  {row['label']}")
    if "sobolev" in r2:
        print(f"\nR2  interior range (base lambda1=0): "
              f"{r2['base_interior_range']:.3f}")
        for p in ("sobolev", "tikhonov"):
            d = r2[p]
            print(f"      {p:8s}: Spearman vs base={d['spearman_vs_base']:+.3f}  "
                  f"range retained={d['range_retained']:.2f}")
    else:
        print(f"\nR2  {r2.get('note','n/a')}")
    print(f"\nR3  conversion sink rank: residual worst={r3['residual_worst']} "
          f"vs phi worst={r3['phi_worst']} (of {r3['num_states']} states)")
    print("\nR4  bootstrap CV (std/|mean|) for top interior states:")
    for row in r4["stability_table"][:6]:
        print(f"      {row['phi_mean']:+.3f} +/- {row['phi_std']:.3f} "
              f"(CV={row['cv']:.2f})  {row['label']}")
    if r5 is not None:
        a = r5["auc"]
        print(f"\nR5  downstream conversion AUC ({r5['conversion_rate']:.0%} "
              f"conversion, {len(r5['seeds'])} splits):")
        for k in ("raw", "raw+sobolev", "raw+ridge", "sobolev", "ridge",
                  "bag_of_states"):
            print(f"      {k:16s} {a[k]['auc_mean']:.4f} +/- {a[k]['auc_std']:.4f}")
    print("================================================")
    print(f"\n[*] wrote {out}")


if __name__ == "__main__":
    main()
