"""Known-answer tests for src/pattern_heuristics.py -- the six named signals."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import main  # noqa: E402
import pattern_heuristics as ph  # noqa: E402

T0 = datetime(2026, 9, 1, 0, 0, 0)


def _df(rows):
    recs = []
    for i, (mins, ins, outs, in_amts, out_amts) in enumerate(rows):
        recs.append({
            "timestamp": T0 + timedelta(minutes=mins),
            "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2",
            "src_port": 50000, "dst_port": 8333,
            "txid": f"tx{i:04d}",
            "input_addresses": ins, "output_addresses": outs,
            "input_amounts": in_amts, "output_amounts": out_amts,
            "geo_country": "IN", "asn": "AS1",
        })
    return pd.DataFrame(recs)


def _signals(df, wallet):
    g = main.build_graph(df)
    wt = main.build_wallet_transfer_graph(df)
    feat = main.compute_wallet_features(df, g).set_index("wallet").loc[wallet].to_dict()
    return {s.name: s for s in ph.detect_all(g, wt, df, wallet, feat)}


def test_all_six_always_returned(df=None):
    df = _df([(0, ["A"], ["B"], [1.0], [1.0])])
    sig = _signals(df, "A")
    assert set(sig) == {"fan_in", "fan_out", "rapid_movement",
                        "amount_splitting", "layering", "circular_flow"}


def test_fan_out_fires_on_a_wide_split():
    outs = [f"o{i}" for i in range(10)]
    df = _df([(0, ["A"], outs, [2.0], [0.2] * 10)])
    sig = _signals(df, "A")
    assert sig["fan_out"].fired
    assert "tx0000" in sig["fan_out"].txids


def test_amount_splitting_fires_only_on_near_equal_parts():
    # near-equal -> fires
    df_eq = _df([(0, ["A"], ["x", "y", "z", "w"], [4.0], [1.0, 1.0, 0.99, 1.01])])
    assert _signals(df_eq, "A")["amount_splitting"].fired
    # one big change output + small peels -> NOT amount_splitting
    df_peel = _df([(0, ["A"], ["x", "y", "z", "c"], [4.0], [0.1, 0.1, 0.1, 3.7])])
    assert not _signals(df_peel, "A")["amount_splitting"].fired


def test_fan_in_fires_on_consolidation():
    ins = [f"s{i}" for i in range(6)]
    df = _df([(0, ins, ["A"], [0.5] * 6, [3.0])])
    sig = _signals(df, "A")
    assert sig["fan_in"].fired
    assert not sig["fan_out"].fired


def test_circular_flow_and_layering_fire_on_a_ring():
    ring = list("ABCDE")
    rows = [(i * 10, [ring[i]], [ring[(i + 1) % 5]], [1.0], [0.99]) for i in range(5)]
    sig = _signals(_df(rows), "A")
    assert sig["circular_flow"].fired
    assert sig["layering"].fired


def test_rapid_movement_fires_on_short_hold():
    df = _df([(0, ["A"], ["B"], [1.0], [1.0]), (2, ["B"], ["C"], [1.0], [0.99])])
    assert _signals(df, "B")["rapid_movement"].fired


def test_nothing_fires_for_an_ordinary_wallet():
    df = _df([
        (0, ["A"], ["B"], [1.0], [0.99]),
        (5000, ["A"], ["C"], [0.5], [0.49]),
    ])
    assert ph.fired_signals(
        main.build_graph(df), main.build_wallet_transfer_graph(df), df, "A",
        main.compute_wallet_features(df, main.build_graph(df)).set_index("wallet").loc["A"].to_dict(),
    ) == []
