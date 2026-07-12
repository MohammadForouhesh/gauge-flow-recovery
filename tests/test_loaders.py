"""Loader tests. Self-contained: each test synthesizes a tiny fixture in the
dataset's on-disk format, loads it, and runs one solve. No downloads needed.
"""
import csv
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poisson_inverse.loaders import (
    load_retailrocket, load_trivago, load_otto)
from poisson_inverse import real_experiments as R


def _rr_fixture(path):
    rows = [["timestamp", "visitorid", "event", "itemid", "transactionid"]]
    t = 1430000000000
    for v in range(120):
        rows.append([t, v, "view", v % 30, ""]); t += 1000
        rows.append([t, v, "addtocart", v % 30, ""]); t += 1000
        if v % 2 == 0:
            rows.append([t, v, "transaction", v % 30, f"tx{v}"]); t += 1000
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def _tv_fixture(path):
    rows = [["user_id", "session_id", "timestamp", "step",
             "action_type", "reference", "impressions"]]
    for s in range(120):
        seq = ["search for destination", "search for item",
               "interaction item info"]
        if s % 2 == 0:
            seq.append("clickout item")
        for i, a in enumerate(seq):
            rows.append([f"u{s%20}", f"s{s}", 1541000000 + s, i + 1,
                         a, str(s % 50), "1|2|3"])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def _otto_fixture(path):
    with open(path, "w") as f:
        t0 = 1660000000000
        for s in range(200):
            evs = [{"aid": s % 40, "ts": t0 + s, "type": "clicks"},
                   {"aid": (s + 1) % 40, "ts": t0 + s + 1, "type": "clicks"}]
            if s % 3 == 0:
                evs.append({"aid": s % 40, "ts": t0 + s + 2, "type": "carts"})
                if s % 2 == 0:
                    evs.append({"aid": s % 40, "ts": t0 + s + 3,
                                "type": "orders"})
            f.write(json.dumps({"session": s, "events": evs}) + "\n")


def _check(ds, expect_states):
    assert ds.num_states == expect_states, \
        f"{ds.name}: expected {expect_states} states, got {ds.num_states}"
    assert len(ds.sessions) > 0
    assert all(s for s in ds.sessions)
    assert ds.conversion_sink_ids and ds.abandon_sink_ids
    # every session ends at a sink
    sink_ids = {k for k, _ in ds.sinks}
    assert all(s[-1] in sink_ids for s in ds.sessions)
    # the pipeline runs and pins the conversion sink at 1
    r = R.recover_potential(ds, lambda1=1.0)
    phi = {row["state"]: row["phi"] for row in r["phi_table"]}
    for c in ds.conversion_sink_ids:
        assert abs(phi[c] - 1.0) < 1e-9


def test_retailrocket(tmp_path=None):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "events.csv")
        _rr_fixture(p)
        ds = load_retailrocket(p, verbose=False)
        _check(ds, expect_states=4)


def test_trivago(tmp_path=None):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "train.csv")
        _tv_fixture(p)
        ds = load_trivago(p, verbose=False)
        _check(ds, expect_states=11)


def test_otto_minimal(tmp_path=None):
    """num_categories=0 -> 5-state graph, no sklearn dependency."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "train.jsonl")
        _otto_fixture(p)
        ds = load_otto(p, num_categories=0, min_item_interactions=1,
                       verbose=False)
        _check(ds, expect_states=5)   # 2 interior + 3 sinks
        # multi-sink: cart sink pinned at 0.6
        cart = [k for k, v in ds.sinks if abs(v - 0.6) < 1e-9]
        assert len(cart) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all loader tests passed")
