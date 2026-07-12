"""
Loaders for three public clickstream corpora, producing the same minimal
state-space structure the Poisson pipeline consumes:

    RealDataset(sessions, num_states, sinks, conversion_sink_ids,
                abandon_sink_ids, state_labels)

Design principle (inherited from the original framework): **states are
event-type-derived, not catalog-derived**. The funnel ordering
(view -> addtocart -> transaction, or search -> interact -> clickout) is a
property of the event-type semantics, which the Helmholtz--Hodge step
recovers from empirical flow. There is no ground-truth potential on real
data, so the recovery-vs-truth experiments do not apply; the relevant
real-data checks are potential *interpretability*, the Tikhonov-vs-Sobolev
contrast on the recovered ordering, blind sink discovery against the known
boundary, and bootstrap stability (see ``real_experiments.py``).

Datasets
--------
RetailRocket : https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
    events.csv  with columns timestamp, visitorid, event, itemid.
    States: view, addtocart, SINK:transaction (phi=1), SINK:abandon (phi=0).

Trivago RecSys 2019 : https://recsys2019data.trivago.com/
    train.csv with columns user_id, session_id, timestamp, step,
    action_type, reference.
    States: the 10 action types (clickout item = conversion, phi=1) plus
    SINK:abandon (phi=0).

OTTO : https://www.kaggle.com/competitions/otto-recommender-system
    JSONL, one session per line: {"session", "events":[{"aid","ts","type"}]}.
    States: {clicks, carts} x {category_0..C} interior, plus three sinks
    SINK:orders (phi=1), SINK:carts (phi=cart_sink_phi, default 0.6),
    SINK:abandon (phi=0). This is the genuine multi-sink regime.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .synthetic import collapse_consecutive


@dataclass
class RealDataset:
    """Minimal state-space dataset consumed by the Poisson pipeline."""
    sessions: List[List[int]]
    num_states: int
    sinks: List[Tuple[int, float]]
    conversion_sink_ids: List[int]
    abandon_sink_ids: List[int]
    state_labels: Dict[int, str] = field(default_factory=dict)
    name: str = "real"

    def summary(self) -> str:
        conv = set(self.conversion_sink_ids)
        n_conv = sum(1 for s in self.sessions if s and s[-1] in conv)
        return (f"{self.name}: {len(self.sessions):,} sessions, "
                f"{self.num_states} states, "
                f"{n_conv:,} conversions "
                f"({100*n_conv/max(1,len(self.sessions)):.2f}%)")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _require_pandas():
    try:
        import pandas as pd  # noqa: F401
        return pd
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "RetailRocket/Trivago loaders require pandas. "
            "Install with: pip install pandas") from e


def _sessionize_by_visitor_and_gap(df, visitor_col, timestamp_col,
                                   gap_minutes=30):
    """Split each visitor's events whenever the inter-event gap exceeds
    ``gap_minutes`` (timestamps in epoch ms). Returns a session-uid Series."""
    pd = _require_pandas()
    df = df.sort_values([visitor_col, timestamp_col], kind="mergesort")
    gap = df.groupby(visitor_col, sort=False)[timestamp_col].diff()
    new_sess = gap.isna() | (gap > gap_minutes * 60 * 1000)
    offset = new_sess.groupby(df[visitor_col]).cumsum().astype(np.int64)
    return df[visitor_col].astype(str) + "::" + offset.astype(str)


def _collapse_states_truncate(state_seq, conv_id, abandon_id,
                              min_len=2):
    """Collapse consecutive duplicates; truncate at first conversion,
    else append the abandon sink. Returns the session or None if too short."""
    if not state_seq:
        return None
    c = [state_seq[0]]
    for s in state_seq[1:]:
        if s != c[-1]:
            c.append(s)
    if len(c) < min_len:
        return None
    if conv_id in c:
        c = c[: c.index(conv_id) + 1]
    else:
        c = c + [abandon_id]
    return c


# ---------------------------------------------------------------------------
# RetailRocket
# ---------------------------------------------------------------------------
def load_retailrocket(events_path: str,
                      fraction: float = 1.0,
                      min_session_length: int = 2,
                      session_gap_minutes: int = 30,
                      seed: int = 42,
                      verbose: bool = True) -> RealDataset:
    """Load RetailRocket events.csv into the 4-state funnel.

    States: ``view`` (0), ``addtocart`` (1), ``SINK:transaction`` (2,
    phi=1), ``SINK:abandon`` (3, phi=0).
    """
    pd = _require_pandas()
    if verbose:
        print(f"[retailrocket] reading {events_path}")
    events = pd.read_csv(
        events_path,
        dtype={"timestamp": np.int64, "visitorid": np.int64,
               "event": str, "itemid": "Int64"},
        usecols=["timestamp", "visitorid", "event", "itemid"])

    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        v = events["visitorid"].unique()
        keep = set(rng.choice(v, size=int(len(v) * fraction),
                              replace=False).tolist())
        events = events[events["visitorid"].isin(keep)]

    labels = ["view", "addtocart", "SINK:transaction", "SINK:abandon"]
    sid = {s: i for i, s in enumerate(labels)}
    conv_id, abandon_id = sid["SINK:transaction"], sid["SINK:abandon"]

    def to_state(ev):
        if ev == "transaction":
            return conv_id
        return sid.get(ev, -1)

    events["state_id"] = events["event"].map(to_state)
    events = events[events["state_id"] >= 0]
    events["session_uid"] = _sessionize_by_visitor_and_gap(
        events, "visitorid", "timestamp", session_gap_minutes)

    sessions: List[List[int]] = []
    for _uid, g in events.sort_values(["session_uid", "timestamp"],
                                      kind="mergesort").groupby(
                                          "session_uid", sort=False):
        sess = _collapse_states_truncate(
            g["state_id"].tolist(), conv_id, abandon_id, min_session_length)
        if sess is not None:
            sessions.append(sess)

    ds = RealDataset(
        sessions=sessions, num_states=len(labels),
        sinks=[(conv_id, 1.0), (abandon_id, 0.0)],
        conversion_sink_ids=[conv_id], abandon_sink_ids=[abandon_id],
        state_labels={i: s for i, s in enumerate(labels)},
        name="RetailRocket")
    if verbose:
        print("  " + ds.summary())
    return ds


# ---------------------------------------------------------------------------
# Trivago
# ---------------------------------------------------------------------------
_TRIVAGO_ACTIONS = [
    "search for destination", "search for item", "search for poi",
    "change of sort order", "filter selection", "interaction item image",
    "interaction item info", "interaction item rating",
    "interaction item deals", "clickout item",   # conversion
]


def load_trivago(train_path: str,
                 fraction: float = 1.0,
                 min_session_length: int = 2,
                 seed: int = 42,
                 verbose: bool = True) -> RealDataset:
    """Load Trivago train.csv into the 11-state action funnel.

    States: the 10 action types (``clickout item`` is the conversion sink,
    phi=1) plus ``SINK:abandon`` (phi=0). Native session ids; ordered by
    step.
    """
    pd = _require_pandas()
    if verbose:
        print(f"[trivago] reading {train_path}")
    events = pd.read_csv(
        train_path,
        dtype={"session_id": str, "timestamp": np.int64,
               "step": np.int64, "action_type": str},
        usecols=["session_id", "timestamp", "step", "action_type"])
    events = events[events["action_type"].isin(_TRIVAGO_ACTIONS)]

    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        s = events["session_id"].unique()
        keep = set(rng.choice(s, size=int(len(s) * fraction),
                              replace=False).tolist())
        events = events[events["session_id"].isin(keep)]

    labels = list(_TRIVAGO_ACTIONS) + ["SINK:abandon"]
    sid = {s: i for i, s in enumerate(labels)}
    conv_id, abandon_id = sid["clickout item"], sid["SINK:abandon"]
    events["state_id"] = events["action_type"].map(sid).astype(np.int64)

    sessions: List[List[int]] = []
    for _sid, g in events.sort_values(["session_id", "step"],
                                      kind="mergesort").groupby(
                                          "session_id", sort=False):
        sess = _collapse_states_truncate(
            g["state_id"].tolist(), conv_id, abandon_id, min_session_length)
        if sess is not None:
            sessions.append(sess)

    ds = RealDataset(
        sessions=sessions, num_states=len(labels),
        sinks=[(conv_id, 1.0), (abandon_id, 0.0)],
        conversion_sink_ids=[conv_id], abandon_sink_ids=[abandon_id],
        state_labels={i: s for i, s in enumerate(labels)},
        name="Trivago")
    if verbose:
        print("  " + ds.summary())
    return ds


# ---------------------------------------------------------------------------
# OTTO (multi-sink)
# ---------------------------------------------------------------------------
def _read_otto_jsonl(path, fraction, seed, max_sessions, verbose):
    rng = np.random.default_rng(seed)
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if fraction < 1.0 and rng.random() > fraction:
                continue
            out.append(json.loads(line))
            if max_sessions and len(out) >= max_sessions:
                break
    if verbose:
        print(f"  [otto] read {len(out):,} sessions from {path}")
    return out


def _covisit_adjacency(sessions_raw, item_to_id, num_items, window=5):
    """Symmetric co-visit counts within a sliding window over each session."""
    import scipy.sparse as sp
    pairs: Counter = Counter()
    for s in sessions_raw:
        ids = [item_to_id.get(int(e["aid"]), -1) for e in s["events"]]
        ids = [x for x in ids if x >= 0]
        for i in range(len(ids)):
            for j in range(i + 1, min(i + 1 + window, len(ids))):
                a, b = ids[i], ids[j]
                if a != b:
                    pairs[(a, b)] += 1
                    pairs[(b, a)] += 1
    if not pairs:
        return sp.csr_matrix((num_items, num_items), dtype=np.float32)
    r, c, v = zip(*[(a, b, cnt) for (a, b), cnt in pairs.items()])
    return sp.coo_matrix((v, (r, c)), shape=(num_items, num_items),
                         dtype=np.float32).tocsr()


def _cluster_items_covisit(W, item_pop, num_categories, seed, verbose):
    """Cluster items into ``num_categories`` via SVD+KMeans on the co-visit
    graph; fall back to log-popularity buckets if scikit-learn is absent."""
    num_items = W.shape[0]
    labels = np.zeros(num_items, dtype=np.int64)
    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.cluster import MiniBatchKMeans
        dim = int(min(64, max(2, num_categories)))
        emb = TruncatedSVD(n_components=dim, random_state=seed
                           ).fit_transform(W.astype(np.float32))
        n = np.linalg.norm(emb, axis=1, keepdims=True)
        n[n == 0] = 1.0
        emb = emb / n
        labels = MiniBatchKMeans(
            n_clusters=int(num_categories), random_state=seed,
            batch_size=4096, n_init=3, max_iter=100
        ).fit_predict(emb).astype(np.int64)
        if verbose:
            print(f"  [otto] co-visit SVD+KMeans -> {num_categories} categories")
    except Exception as exc:  # pragma: no cover
        if verbose:
            print(f"  [otto] sklearn unavailable ({exc}); "
                  f"log-popularity bucketing")
        pop = np.log1p(item_pop.astype(np.float64))
        if pop.max() > pop.min():
            edges = np.quantile(pop, np.linspace(0, 1, num_categories + 1)[1:-1])
            labels = np.digitize(pop, edges).astype(np.int64)
    np.clip(labels, 0, num_categories - 1, out=labels)
    return labels


def load_otto(train_path: str,
              fraction: float = 1.0,
              num_categories: int = 50,
              cart_sink_phi: float = 0.6,
              min_item_interactions: int = 5,
              min_session_length: int = 2,
              co_visit_window: int = 5,
              max_sessions: int = 0,
              seed: int = 42,
              verbose: bool = True) -> RealDataset:
    """Load OTTO JSONL into the {clicks,carts} x category multi-sink funnel.

    Interior states: ``clicks`` x category and ``carts`` x category, where
    categories (``num_categories`` of them, plus an ``unknown`` bucket) are
    co-visit clusters. Terminals: ``SINK:orders`` (phi=1), ``SINK:carts``
    (phi=``cart_sink_phi``), ``SINK:abandon`` (phi=0). A session is routed to
    its best achieved outcome (order > cart > abandon).

    Set ``num_categories=0`` for a minimal 5-state graph
    (clicks, carts, and the three sinks) with no co-visit clustering and no
    scikit-learn dependency.
    """
    raw = _read_otto_jsonl(train_path, fraction, seed, max_sessions, verbose)
    if not raw:
        raise RuntimeError("no OTTO sessions read")

    # Item vocabulary (min-interaction filter).
    cnt: Counter = Counter()
    for s in raw:
        for e in s["events"]:
            cnt[int(e["aid"])] += 1
    keep = sorted(a for a, c in cnt.items() if c >= min_item_interactions)
    item_to_id = {a: i for i, a in enumerate(keep)}
    num_items = len(item_to_id)
    if verbose:
        print(f"  [otto] {num_items:,} items pass >= "
              f"{min_item_interactions} interactions")

    # Categories.
    if num_categories and num_categories > 0 and num_items > 0:
        item_pop = np.array([cnt[a] for a in keep], dtype=np.float64)
        W = _covisit_adjacency(raw, item_to_id, num_items, co_visit_window)
        item_cat = _cluster_items_covisit(W, item_pop, num_categories,
                                          seed, verbose)
        C_plus = num_categories + 1                 # +1 unknown bucket
        UNKNOWN = num_categories
    else:
        item_cat = None
        C_plus = 1
        UNKNOWN = 0

    def cat_of(aid):
        if item_cat is None:
            return 0
        iid = item_to_id.get(int(aid), -1)
        return UNKNOWN if iid < 0 else int(item_cat[iid])

    ORDER_SINK = 2 * C_plus
    CART_SINK = 2 * C_plus + 1
    ABANDON_SINK = 2 * C_plus + 2
    num_states = 2 * C_plus + 3

    def interior_state(etype, aid):
        c = cat_of(aid)
        if etype == "clicks":
            return c
        if etype == "carts":
            return C_plus + c
        return None

    sessions: List[List[int]] = []
    n_order = n_cart = n_abandon = 0
    for s in raw:
        evs = sorted(s["events"], key=lambda e: e["ts"])
        seq, has_order, has_cart = [], False, False
        for e in evs:
            t = e["type"]
            if t == "orders":
                has_order = True
                continue
            if t == "carts":
                has_cart = True
            st = interior_state(t, e["aid"])
            if st is not None:
                seq.append(st)
        if not seq:
            continue
        c = [seq[0]]
        for st in seq[1:]:
            if st != c[-1]:
                c.append(st)
        if len(c) < min_session_length:
            continue
        if has_order:
            term = ORDER_SINK; n_order += 1
        elif has_cart:
            term = CART_SINK; n_cart += 1
        else:
            term = ABANDON_SINK; n_abandon += 1
        sessions.append(c + [term])

    labels = {}
    for c in range(C_plus):
        labels[c] = f"clicks:cat_{c}" if item_cat is not None else "clicks"
        labels[C_plus + c] = f"carts:cat_{c}" if item_cat is not None else "carts"
    labels[ORDER_SINK] = "SINK:orders"
    labels[CART_SINK] = "SINK:carts"
    labels[ABANDON_SINK] = "SINK:abandon"

    ds = RealDataset(
        sessions=sessions, num_states=num_states,
        sinks=[(ORDER_SINK, 1.0), (CART_SINK, float(cart_sink_phi)),
               (ABANDON_SINK, 0.0)],
        conversion_sink_ids=[ORDER_SINK], abandon_sink_ids=[ABANDON_SINK],
        state_labels=labels, name="OTTO")
    if verbose:
        print(f"  [otto] routing: {n_order:,} orders / {n_cart:,} carts / "
              f"{n_abandon:,} abandon")
        print("  " + ds.summary())
    return ds
