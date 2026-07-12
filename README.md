# Gauge-Invariant, Parameter-Insensitive Regularization for Potential Recovery from Flow on Directed Graphs

Reproducible code for the paper. The package recovers a latent scalar
**potential** from observed **directed flow** by solving a Dirichlet-anchored
discrete Poisson problem on an extracted DAG, and contrasts two regularizers:

- **graph-Sobolev** — penalizes the conductance-weighted Dirichlet energy
  `phi^T L_G phi = sum_{(u,v)} w_uv (phi_v - phi_u)^2` (**gauge-invariant**:
  `L_G 1 = 0`, penalizes differences not amplitude), and
- **Tikhonov / ridge** — penalizes magnitude `||phi||^2` (**gauge-dependent**:
  shrinks toward a boundary-induced origin).

The headline is **parameter-insensitivity**: on a planted-ground-truth
instrument the graph-Sobolev estimate holds rank correlation `+0.81` across four
orders of magnitude in `lambda1`, whereas ridge **inverts** the ordering
(`-0.42`) for every `lambda1 > 0`. Everything below regenerates the exact numbers
in the paper's tables and figures from a fixed seed.

---

## 1. Installation

```bash
pip install -r requirements.txt
```

| Requirement | Needed for |
|---|---|
| Python ≥ 3.9, `numpy`, `scipy` | core solver + synthetic experiments |
| `matplotlib` | figures |
| `pandas` | RetailRocket, Trivago loaders |
| `scikit-learn` | OTTO co-visit clustering (falls back to popularity buckets if absent) |
| `torch` ≥ 2.0 | *optional* — directed-GNN oversmoothing baseline (`run_gnn.py`) |

No GPU is required; every experiment runs on a laptop CPU.

---

## 2. Repository layout

```
poisson_inverse/
  synthetic.py         planted-ground-truth funnel generator + chain builder
  core.py              pipeline: empirical flow -> dominance filter -> DAG
                       extraction (HHD or topological sort) -> divergence ->
                       gauge-stable Poisson solve -> Poisson-residual sink
                       discovery; nilpotency / reachability diagnostics
  experiments.py       synthetic experiments E1-E7 + solver scaling & recovery
  loaders.py           RetailRocket / Trivago / OTTO -> event-type state spaces
  real_experiments.py  real-data checks R1-R5 (no ground truth)
scripts/
  run_experiments.py   synthetic E1-E7      -> results_synthetic.json
  run_real_data.py     real R1-R5           -> results_<dataset>.json
  run_scaling.py       solver scaling + recovery at 10^4 nodes -> scaling_results.json
  run_gnn.py           directed-GNN oversmoothing baseline + gauge-centering (needs torch)
  make_figures.py      Figures 1-3 (synthetic) as PDFs
  make_real_figures.py Figures 4-6 (real data) as PDFs
tests/
  test_core.py         fast solver + claim checks (incl. CG == dense agreement)
  test_loaders.py      loader checks on self-contained fixtures
results/               committed reference outputs (the numbers behind the paper)
```

---

## 3. Quickstart

```bash
# Regenerate all synthetic numbers (seed 42, ~90s) into the reference folder
python scripts/run_experiments.py --seed 42 --out results/results_synthetic.json

# Run the fast test suite
python tests/test_core.py && python tests/test_loaders.py
```

The `results/` directory holds the committed reference JSONs. Re-running a script
with `--out results/<file>.json` overwrites the reference in place, so
`git diff results/` (or a manual compare) shows exactly what, if anything,
changed. Synthetic and scaling numbers are seed-deterministic; real-data numbers
depend only on the corpus and the flags below.

---

## 4. Reproducing every result

Each paper artifact maps to one command and one output file:

| Paper artifact | Command | Output → |
|---|---|---|
| Table 1 `tab:sweep`, Fig. 1 (regularizer sweep) | `run_experiments.py` (E1) / `make_figures.py` | `results/results_synthetic.json` / `figs/fig1_*.pdf` |
| Table 2 `tab:ablation` (extraction ablation) | `run_experiments.py` (E6) | `results/results_synthetic.json` |
| Table 6 `tab:robustness` (12-variant robustness) | `run_experiments.py` (E7) | `results/results_synthetic.json` |
| Chain range (Thm 2), Fig. 3 | `run_experiments.py` (E2) / `make_figures.py` | `results/results_synthetic.json` / `figs/fig3_*.pdf` |
| Sink discovery `tab:sink` (E3) | `run_experiments.py` (E3) | `results/results_synthetic.json` |
| Nilpotency `m=7` (E4), Hodge negative (E5) | `run_experiments.py` | `results/results_synthetic.json` |
| Fig. 2 (recovery scatter) | `make_figures.py` | `figs/fig2_*.pdf` |
| Real-data `tab:real-contrast`, Fig. 4-6 (R1-R4) | `run_real_data.py <ds>` / `make_real_figures.py` | `results/results_<ds>.json` |
| Downstream `tab:downstream` (R5) | `run_real_data.py otto` (R5) | `results/results_otto.json` |
| Scaling `tab:scaling` + recovery at 10^4 nodes | `run_scaling.py` | `results/scaling_results.json` |
| GNN oversmoothing + gauge-centering (§2, Table 6 GNN rows) | `run_gnn.py --train <otto>` | stdout (not folder-backed) |

### 4.1 Synthetic experiments (E1–E7)

```bash
python scripts/run_experiments.py --seed 42 --out results/results_synthetic.json
```

Generates the planted funnel (`|V| = 277`, five sinks, harmonic ground truth),
then runs:

- **E1 regularizer sweep** — Spearman / Pearson / NDCG@5 of the recovered
  potential vs the planted `phi_true`, for both penalties across
  `lambda1 in {0, 1e-3, ..., 10}`, on the **determined interior** (well-visited
  states with a directed path to the boundary; see §7).
- **E2 chain dynamic range** — exact range preservation on a length-30 chain.
- **E3 blind sink discovery** — Poisson-residual vs potential ranking of sinks.
- **E4 nilpotency** — `||A^m||_F = 0` at the predicted index.
- **E5 gradient preservation** — honest negative: the HHD gradient alone does
  not recover the planted direction.
- **E6 extraction ablation** — HHD vs two topological sorts.
- **E7 regularizer robustness** — the Sobolev-minus-ridge margin across 12
  instrument variants (5 seeds + 4 structural axes).

### 4.2 Figures

```bash
python scripts/make_figures.py --seed 42 --outdir ../paper-LoG/figs        # Fig. 1-3
python scripts/make_real_figures.py --outdir ../paper-LoG/figs             # Fig. 4-6 (after §4.3)
```

(`--outdir` defaults elsewhere; point it at the paper's `figs/` directory.)

### 4.3 Real-data experiments (R1–R5)

No clickstream corpus carries a ground-truth potential, so the real-data checks
validate claims that need no planted field. **Get the data** (one file each):

| Dataset | File | Source |
|---|---|---|
| RetailRocket | `events.csv` | kaggle.com/datasets/retailrocket/ecommerce-dataset |
| Trivago | `train.csv` | recsys2019data.trivago.com (free signup) |
| OTTO | `train.jsonl` | kaggle.com/competitions/otto-recommender-system |

**Run** (each writes `results/results_<dataset>.json` and prints a summary):

```bash
python scripts/run_real_data.py retailrocket --events /path/to/events.csv \
    --out results/results_retailrocket.json
python scripts/run_real_data.py trivago --train /path/to/train.csv \
    --out results/results_trivago.json
python scripts/run_real_data.py otto --train /path/to/train.jsonl \
    --num_categories 50 --max_sessions 200000 --out results/results_otto.json
```

Checks: **R1** recovered potential (interpretability), **R2** regularizer
contrast (range retention + rank agreement vs the unregularized base),
**R3** residual sink discovery against the known boundary, **R4** bootstrap
stability, **R5** downstream session-level conversion prediction (Sobolev vs
ridge potential features; leakage-guarded — features use only interior states,
the label is the terminal sink, and the potential is fit on the training split
only). Useful flags: `--fraction`, `--lambda1`, `--rho`/`--tau`/`--top_k`,
`--n_boot`, `--no_downstream`; for OTTO, `--num_categories 0` gives a minimal
5-state graph with no scikit-learn dependency and `--max_sessions` caps memory.
On small graphs (RetailRocket, 2 interior states) the R2/R5 contrast is muted;
it is clearest on OTTO's 105-state multi-sink graph. If a run reports an empty
DAG, lower `--tau` (e.g. `0.3`).

### 4.4 Solver scaling + recovery at 10^4 nodes

```bash
cd results && python ../scripts/run_scaling.py    # writes scaling_results.json here
```

Two parts, both in `results/scaling_results.json`: (a) **timing** — sparse
conjugate gradients vs the dense eigensolve on layered DAGs of growing interior
size (CG matches the dense solve to `1e-9` and scales past `4x10^4` nodes where
the dense solve exhausts memory); (b) **recovery** — the regularizer contrast on
a planted funnel with `|V| = 10,202`, where graph-Sobolev recovers Spearman
`+0.94` and ridge inverts to `-0.18` via the gauge-stable CG path. (Timings are
wall-clock and machine-dependent; the recovery correlations are seed-deterministic.)

### 4.5 Directed-GNN oversmoothing baseline (optional, needs `torch`)

```bash
python scripts/run_gnn.py --train /path/to/otto/train.jsonl --max_sessions 200000
```

Trains a directed GCN (separate in/out aggregation) of increasing depth on the
OTTO conversion task, pooling node embeddings over the same visited interior
states as the potential features. It reports (a) pure-propagation Dirichlet-energy
collapse, (b) test AUC + node energy vs depth for a vanilla GCN (oversmooths),
and (c) the **gauge-centered** variant — removing the constant mode
`u_0 ∝ d^{1/2}` at each layer, the per-layer analogue of `L_G 1 = 0` — which
holds AUC flat with depth (recovering PairNorm / Dirichlet-energy-constrained
GNNs from the gauge principle). Prints to stdout; not part of the seed-backed
`results/` set.

---

## 5. Expected numbers (seed 42)

| Experiment | Result |
|---|---|
| E1 sweep (`lambda1=1`, HHD, n=95) | Sobolev Spearman **+0.81** / NDCG@5 **0.999**; ridge **−0.42** / 0.756 |
| E1 across `lambda1 in [1e-3, 10]` | Sobolev Spearman constant at **+0.807**; ridge negative throughout |
| E2 chain range (`lambda1=1`) | Sobolev Δφ **0.968**; ridge **0.536** |
| E3 sink discovery | residual ranks all 5 sinks in top **7** of 277; potential ranking worst **108** |
| E4 nilpotency | `||A^m||_F = 0` at **m = 7** |
| E5 gradient (honest negative) | HHD gradient cosine **−0.06** |
| E6 ablation margins (Sobolev − ridge) | HHD **+1.23**, topo-visit **+1.32**, topo-net-inflow **+1.52** |
| E7 robustness | margin positive in **12/12** variants, both extractions |
| Scaling recovery (`\|V\|=10,202`) | Sobolev Spearman **+0.94**; ridge **−0.18** (CG ~0.08 s) |
| Real R2 (OTTO range retained) | Sobolev **28%**, ridge **0.2%** |
| Real R5 (OTTO downstream AUC) | Sobolev-φ **0.836** vs ridge-φ **0.717**; raw+Sobolev **0.859** |
| GNN depth sweep (OTTO) | vanilla **0.86 → 0.81** (2→32 layers); gauge-centered flat **0.85** |

---

## 6. Minimal API usage

```python
from poisson_inverse import (generate_funnel, empirical_flow, dominance_filter,
    hhd_dag_extract, top_k_prune, divergence, solve_poisson)

f   = generate_funnel(seed=42)                       # planted harmonic ground truth
F   = empirical_flow(f.sessions, f.num_states)       # empirical directed flow
phi0, retained, _ = hhd_dag_extract(F, dominance_filter(F, rho=2.0), f.num_states)
dag = top_k_prune(retained, top_k=10)                # acyclic support
b   = divergence(F, f.num_states)                    # Poisson right-hand side
phi, residual = solve_poisson(dag, f.num_states, dict(f.sinks), b,
                              lambda1=1.0, penalty="sobolev")  # or "tikhonov"
```

`solve_poisson(..., method=...)` selects the linear solver: `"auto"` (default:
dense for small interiors, sparse CG for large), `"cholesky"` (dense), or `"cg"`.

---

## 7. Method notes and caveats

- **Gauge-stable solve.** Theorem 1 makes the reduced operator `M` SPD *when
  every interior node has a directed path to the boundary*. Real extractions
  violate this (many low-traffic states never reach a sink), so `M` is only
  positive semidefinite. The dense path therefore uses a symmetric eigensolve
  that inverts only the nonzero spectrum (the minimum-norm solution), and the CG
  path runs floor-free from `x0 = 0`; both keep the gauge mode exact where a
  Cholesky-with-floor solve would amplify roundoff. Cholesky suffices only in the
  genuinely SPD case (e.g. the fully-connected scaling DAGs).
- **Determined interior + tie-robust scoring.** Recovery is scored on
  well-visited states that also reach the boundary; nodes without a path are
  undetermined by the Dirichlet data and excluded. Rank correlations round the
  estimate to numerical tolerance so the saturated near-zero cluster ties
  deterministically rather than by roundoff.
- **Weighted energy.** The graph-Sobolev penalty uses the conductance-weighted
  Dirichlet energy (each edge contributes `w_uv (phi_v - phi_u)^2`); the
  unweighted incidence Laplacian is the `w == 1` special case.
- **Preservation, not signal.** The penalty guarantees *preservation and
  robustness*, not signal. Recovery *magnitude* is regime-dependent (compressed
  at low abandonment); the regularizer's role — Sobolev preserves whatever
  ordering the unregularized solve attains, ridge degrades it — is invariant.
- **No ground truth on real data.** Real corpora validate internal consistency
  and the regularizer contrast, not recovery accuracy; the recovery-vs-truth
  numbers come from the synthetic instrument.

---

## 8. Tests

```bash
python tests/test_core.py      # solver, gauge invariance, chain range,
                               # parameter-insensitivity, CG==dense, nilpotency
python tests/test_loaders.py   # loader fixtures for all three corpora
```

(Run with `pytest tests/` if `pytest` is installed.)
