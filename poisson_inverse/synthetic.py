"""
Multi-branch, multi-sink synthetic conversion funnel with a *planted*
harmonic ground-truth potential.

This is the controlled instrument used throughout the paper. It plants a
known scalar field ``phi_true`` (the harmonic absorption probability into
the conversion sinks) and samples sessions from the planted transition
matrix, so that the recovered potential can be scored against ground
truth -- something no real clickstream corpus provides.

The generator is deterministic given ``seed``. Only the topology, the
planted transition matrix ``P``, the planted ``phi_true``, and the sampled
sessions are produced; no item catalog is built (this paper is about the
state-space inverse problem, not item recommendation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp


def collapse_consecutive(seq: List[int]) -> List[int]:
    """Collapse consecutive duplicate states (reduction operator D)."""
    if not seq:
        return seq
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out


@dataclass
class SyntheticFunnel:
    """A generated funnel instance with planted ground truth."""
    sessions: List[List[int]]
    P: np.ndarray                       # (N, N) planted row-stochastic transitions
    phi_true: np.ndarray                # (N,) planted harmonic potential
    num_states: int
    conversion_sink_ids: List[int]
    abandon_sink_ids: List[int]
    sinks: List[Tuple[int, float]]      # Dirichlet (state, value) pairs
    branch_of_state: np.ndarray         # (N,) branch id, -1 for source/shared/abandon
    sink_entropy: float                 # H(terminal-sink distribution), bits


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
def _branch_probabilities(num_branches: int, entropy_param: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Branch-preference distribution with controllable entropy.

    ``entropy_param`` in [0, 1] maps to a Dirichlet concentration
    ``alpha = 1 + 9 * entropy_param``; 1.0 yields a near-uniform
    (max-entropy multi-sink) regime.
    """
    alpha = 1.0 + entropy_param * 9.0
    return rng.dirichlet(alpha * np.ones(num_branches))


def _build_topology(num_branches: int, shared_top_states: int,
                    states_per_branch: int):
    """Lay out state ids.

    Layout::

        0                               source
        1..shared_top_states            shared upper states
        per branch b:
            base = 1 + shared_top_states + b * (states_per_branch + 1)
            base + 0 .. base + states_per_branch - 1   branch interior
            base + states_per_branch                   conversion sink
        last state                      abandon sink
    """
    source = 0
    shared_end = 1 + shared_top_states              # exclusive
    branch_block = states_per_branch + 1
    branches_start = shared_end
    branches_end = branches_start + num_branches * branch_block
    abandon = branches_end
    num_states = branches_end + 1

    conversion_sinks: List[int] = []
    branch_of_state = -np.ones(num_states, dtype=np.int64)
    for b in range(num_branches):
        base = branches_start + b * branch_block
        for k in range(states_per_branch):
            branch_of_state[base + k] = b
        sink = base + states_per_branch
        conversion_sinks.append(sink)
        branch_of_state[sink] = b
    return num_states, source, shared_end, conversion_sinks, abandon, branch_of_state


def _build_transition_weights(num_states, source, shared_end, conversion_sinks,
                              abandon, branch_of_state, states_per_branch,
                              branch_probs, abandon_rate, rng) -> sp.csr_matrix:
    """Latent transition weights T (the data-generating process)."""
    num_branches = len(conversion_sinks)
    rows, cols, vals = [], [], []

    def add(i, j, w):
        if w <= 0:
            return
        rows.append(i); cols.append(j); vals.append(float(w))

    shared_ids = list(range(1, shared_end))
    for s in shared_ids:
        add(source, s, 1.0 + 0.1 * rng.random())

    branch_block = states_per_branch + 1
    branch_entry = [shared_end + b * branch_block for b in range(num_branches)]
    for k, s in enumerate(shared_ids):
        if k + 1 < len(shared_ids):
            add(s, shared_ids[k + 1], 0.4 + 0.1 * rng.random())
        for b in range(num_branches):
            add(s, branch_entry[b], 2.0 * branch_probs[b] + 0.02 * rng.random())

    for b in range(num_branches):
        interior = [shared_end + b * branch_block + k
                    for k in range(states_per_branch)]
        sink = conversion_sinks[b]
        for d, s in enumerate(interior):
            if d + 1 < len(interior):
                add(s, interior[d + 1], 1.0 + 0.05 * rng.random())
            if d + 3 < len(interior) and rng.random() < 0.25:
                add(s, interior[d + 3], 0.2 + 0.05 * rng.random())
            sink_w = 0.05 + 0.55 * (d / max(1, len(interior) - 1))
            add(s, sink, sink_w)

    # Abandon arcs with depth-decay (shallow states abandon more).
    n_shared = max(1, len(shared_ids))
    for k, s in enumerate(shared_ids):
        depth_frac = (k + 1) / n_shared
        add(s, abandon, abandon_rate * (1.0 - 0.3 * depth_frac))
    add(source, abandon, abandon_rate)
    for b in range(num_branches):
        interior = [shared_end + b * branch_block + k
                    for k in range(states_per_branch)]
        for d, s in enumerate(interior):
            depth_frac = d / max(1, len(interior) - 1)
            add(s, abandon, abandon_rate * (1.0 - 0.7 * depth_frac))

    return sp.csr_matrix((vals, (rows, cols)),
                         shape=(num_states, num_states), dtype=np.float64)


def _row_normalize(T: sp.csr_matrix) -> np.ndarray:
    T = T.toarray().astype(np.float64)
    rs = T.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return T / rs


def _build_phi_true_harmonic(P, num_states, conversion_sinks, abandon) -> np.ndarray:
    """phi_true[u] = P(reach a conversion sink before abandon | start u).

    Harmonic extension of the Dirichlet boundary {conversion=1, abandon=0}
    under the planted transitions P, via fixed-point iteration.
    """
    phi = np.zeros(num_states, dtype=np.float64)
    for s in conversion_sinks:
        phi[s] = 1.0
    phi[abandon] = 0.0
    sinks_all = set(conversion_sinks) | {abandon}
    for _ in range(5000):
        phi_new = P @ phi
        for s in sinks_all:
            phi_new[s] = 1.0 if s in conversion_sinks else 0.0
        if np.linalg.norm(phi_new - phi) < 1e-12:
            phi = phi_new
            break
        phi = phi_new
    return phi


def _generate_sessions(P, source, conversion_sinks, abandon,
                       num_sessions, max_steps, rng):
    """Sample sessions by running the Markov chain P from source."""
    sink_set = set(conversion_sinks) | {abandon}
    sessions: List[List[int]] = []
    terminals = np.zeros(num_sessions, dtype=np.int64)
    state_ids = np.arange(P.shape[0])
    for s in range(num_sessions):
        path = [source]
        cur = source
        for _ in range(max_steps):
            row = P[cur]
            if row.sum() <= 0:
                break
            nxt = int(rng.choice(state_ids, p=row))
            path.append(nxt)
            cur = nxt
            if nxt in sink_set:
                break
        sessions.append(path)
        terminals[s] = path[-1]
    return sessions, terminals


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_funnel(num_branches: int = 5,
                    states_per_branch: int = 50,
                    shared_top_states: int = 20,
                    num_sessions: int = 50_000,
                    max_session_steps: int = 30,
                    sink_entropy_param: float = 1.0,
                    abandon_rate: float = 0.05,
                    seed: int = 42) -> SyntheticFunnel:
    """Build a synthetic funnel with a planted harmonic ground truth.

    The default operating point (``abandon_rate=0.05``,
    ``shared_top_states=20``) yields ``|V| = 277`` states with five
    conversion sinks, matching the configuration reported in the paper.

    The session walks and ``phi_true`` are produced *before* any auxiliary
    structure, so they are byte-for-byte reproducible for a given seed.
    """
    rng = np.random.default_rng(seed)

    (num_states, source, shared_end, conversion_sinks, abandon,
     branch_of_state) = _build_topology(num_branches, shared_top_states,
                                         states_per_branch)
    branch_probs = _branch_probabilities(num_branches, sink_entropy_param, rng)
    T = _build_transition_weights(
        num_states, source, shared_end, conversion_sinks, abandon,
        branch_of_state, states_per_branch, branch_probs, abandon_rate, rng)
    P = _row_normalize(T)
    phi_true = _build_phi_true_harmonic(P, num_states, conversion_sinks, abandon)
    sessions, terminals = _generate_sessions(
        P, source, conversion_sinks, abandon, num_sessions, max_session_steps, rng)
    sessions = [collapse_consecutive(s) for s in sessions]

    # Empirical terminal-sink distribution and its Shannon entropy (bits).
    pi = np.zeros(num_branches, dtype=np.float64)
    for t in terminals:
        if t in conversion_sinks:
            pi[int(branch_of_state[t])] += 1
    tot = pi.sum()
    if tot > 0:
        pi /= tot
    H = float(-np.sum(pi[pi > 0] * np.log2(pi[pi > 0])))

    sinks = [(s, 1.0) for s in conversion_sinks] + [(abandon, 0.0)]
    return SyntheticFunnel(
        sessions=sessions, P=P, phi_true=phi_true, num_states=num_states,
        conversion_sink_ids=list(conversion_sinks), abandon_sink_ids=[abandon],
        sinks=sinks, branch_of_state=branch_of_state, sink_entropy=H)


def build_chain(length: int = 30, sink_value: float = 1.0):
    """A directed chain  src -> v1 -> ... -> v_{length} (sink), used for the
    dynamic-range experiment.

    Only the sink is pinned (Dirichlet); the source and interior float, so
    a magnitude penalty has a shrinkage degree of freedom to collapse. The
    divergence injects unit flow at the source and absorbs it at the sink::

        b[src] = -1,   b[sink] = +1,   b[interior] = 0

    The harmonic solution is the linear ramp with interior dynamic range 1.

    Returns (dag_edges, num_states, sink_map, b).
    """
    num_states = length + 2                       # source + length interior + sink
    src, sink = 0, num_states - 1
    dag_edges = [(i, i + 1, 1.0) for i in range(num_states - 1)]
    sink_map = {sink: sink_value}                 # only the sink is pinned
    b = np.zeros(num_states)
    b[src] = -1.0
    b[sink] = +1.0
    return dag_edges, num_states, sink_map, b
