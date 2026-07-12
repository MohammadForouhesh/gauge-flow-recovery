#!/usr/bin/env python3
"""Run every experiment in the paper and write results to results.json.

Usage:
    python scripts/run_experiments.py [--seed 42] [--out results.json]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poisson_inverse.synthetic import generate_funnel
from poisson_inverse import experiments as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_sessions", type=int, default=50_000)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[*] generating default funnel (seed={args.seed}) ...")
    funnel = generate_funnel(num_sessions=args.num_sessions, seed=args.seed)
    print(f"    |V| = {funnel.num_states}, "
          f"sinks = {len(funnel.conversion_sink_ids)}, "
          f"H(pi) = {funnel.sink_entropy:.2f} bits")

    results = {"config": {"seed": args.seed, "num_sessions": args.num_sessions,
                          "num_states": funnel.num_states,
                          "sink_entropy_bits": funnel.sink_entropy}}

    print("[E1] regularizer sweep (Sobolev vs Tikhonov) ...")
    results["E1_regularizer_sweep"] = E.regularizer_sweep(funnel)

    print("[E2] chain dynamic range ...")
    results["E2_chain_dynamic_range"] = E.chain_dynamic_range(length=30)

    print("[E3] blind sink discovery ...")
    results["E3_sink_discovery"] = E.sink_discovery(funnel)
    print("     cross-regime ...")
    results["E3_cross_regime"] = E.sink_discovery_cross_regime(seed=args.seed)

    print("[E4] nilpotency ...")
    results["E4_nilpotency"] = E.nilpotency(funnel)

    print("[E5] gradient-direction preservation ...")
    results["E5_gradient_preservation"] = E.gradient_preservation(funnel)

    print("[E6] extraction ablation (HHD vs topological sort) ...")
    results["E6_extraction_ablation"] = E.extraction_ablation(funnel)

    print("[E7] regularizer robustness (seeds x instrument configs) ...")
    results["E7_regularizer_robustness"] = E.regularizer_robustness(
        base_seed=args.seed, num_sessions=args.num_sessions)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] wrote {args.out} in {time.time() - t0:.1f}s")

    # Console summary
    s = results
    e1 = s["E1_regularizer_sweep"]
    sob1 = next(r for r in e1["sobolev"] if r["lambda1"] == 1.0)
    tik1 = next(r for r in e1["tikhonov"] if r["lambda1"] == 1.0)
    base = next(r for r in e1["sobolev"] if r["lambda1"] == 0.0)
    print("\n==================== SUMMARY ====================")
    print(f"E1  baseline (lambda1=0)  Spearman={base['spearman']:+.3f}  "
          f"Pearson={base['pearson']:+.3f}  (n_well={e1['n_well']})")
    print(f"E1  Sobolev  (lambda1=1)  Spearman={sob1['spearman']:+.3f}  "
          f"Pearson={sob1['pearson']:+.3f}")
    print(f"E1  Tikhonov (lambda1=1)  Spearman={tik1['spearman']:+.3f}  "
          f"Pearson={tik1['pearson']:+.3f}")
    e2 = s["E2_chain_dynamic_range"]
    sob_dphi = next(r for r in e2["sobolev"] if r["lambda1"] == 1.0)["delta_phi"]
    tik_dphi = next(r for r in e2["tikhonov"] if r["lambda1"] == 1.0)["delta_phi"]
    print(f"E2  chain Delta-phi @ lambda1=1   Sobolev={sob_dphi:.3f}  "
          f"Tikhonov={tik_dphi:.3f}")
    e3 = s["E3_sink_discovery"]
    print(f"E3  residual sink ranks {e3['residual_ranks']} (worst {e3['residual_worst']})"
          f"  vs direct-phi worst {e3['phi_worst']}")
    e4 = s["E4_nilpotency"]
    print(f"E4  nilpotency index m={e4['nilpotency_index']}  "
          f"||A^m||_F={e4['frobenius_at_index']:.1e}")
    e5 = s["E5_gradient_preservation"]
    print(f"E5  cosine(HHD phi0 grad)={e5['cosine_hhd_gradient']:+.3f}  "
          f"cosine(recovered phi grad)={e5['cosine_recovered_phi']:+.3f} "
          f"(n_edges={e5['n_edges']})")
    e6 = s["E6_extraction_ablation"]
    print("E6  extraction ablation (Sobolev vs Tikhonov Spearman):")
    for r in e6["rows"]:
        print(f"      {r['extraction']:20s} edges={r['n_edges']:4d} n={r['n_scored']:3d}  "
              f"Sob={r['sobolev']['spearman']:+.3f}  "
              f"Tik={r['tikhonov']['spearman']:+.3f}  "
              f"margin={r['margin_spearman']:+.3f}")
    e7 = s["E7_regularizer_robustness"]
    print(f"E7  regularizer robustness ({e7['n_variants']} instrument variants, "
          f"Sobolev-Tikhonov Spearman margin @ lambda1={e7['lambda1']}):")
    for ext, sm in e7["summary"].items():
        print(f"      {ext:20s} margin={sm['margin_mean']:+.3f}+/-{sm['margin_std']:.3f}  "
              f"min={sm['margin_min']:+.3f}  positive={sm['n_positive']}/{sm['n_variants']}  "
              f"(Sob={sm['sobolev_mean']:+.3f}+/-{sm['sobolev_std']:.3f})")
    print("================================================")


if __name__ == "__main__":
    main()
