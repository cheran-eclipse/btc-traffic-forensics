"""Known-answer tests for the Night 2 multi-hop features in src/main.py."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main  # noqa: E402

T0 = datetime(2026, 9, 1, 0, 0, 0)


def _df(rows):
    """rows: (minutes_offset, [inputs], [outputs]) -> transactions DataFrame."""
    recs = []
    for i, (mins, ins, outs) in enumerate(rows):
        recs.append(
            {
                "timestamp": T0 + timedelta(minutes=mins),
                "src_ip": "10.0.0.1",
                "dst_ip": "10.0.0.2",
                "src_port": 50000,
                "dst_port": 8333,
                "txid": f"tx{i:04d}",
                "input_addresses": ins,
                "output_addresses": outs,
                "input_amounts": [1.0] * len(ins),
                "output_amounts": [1.0 / len(outs)] * len(outs),
                "geo_country": "IN",
                "asn": "AS1",
            }
        )
    return pd.DataFrame(recs)


def test_min_return_cycle_hops_finds_a_ring():
    # A -> B -> C -> A
    df = _df([(0, ["A"], ["B"]), (10, ["B"], ["C"]), (20, ["C"], ["A"])])
    wt = main.build_wallet_transfer_graph(df)
    assert main.min_return_cycle_hops(wt, "A") == 3
    assert main.min_return_cycle_hops(wt, "B") == 3


def test_min_return_cycle_hops_zero_when_no_cycle():
    df = _df([(0, ["A"], ["B"]), (10, ["B"], ["C"])])
    wt = main.build_wallet_transfer_graph(df)
    assert main.min_return_cycle_hops(wt, "A") == 0
    assert main.min_return_cycle_hops(wt, "C") == 0


def test_linear_chain_length_counts_the_whole_chain():
    # A -> B -> C -> D -> E, each hop a single 1-in/1-out wallet
    df = _df([
        (0, ["A"], ["B"]), (10, ["B"], ["C"]),
        (20, ["C"], ["D"]), (30, ["D"], ["E"]),
    ])
    wt = main.build_wallet_transfer_graph(df)
    assert main.linear_chain_length(wt, "C") == 5
    assert main.linear_chain_length(wt, "A") == 5


def test_linear_chain_length_breaks_at_a_branch():
    # B fans out to C and D -> B is not a clean pass-through node
    df = _df([(0, ["A"], ["B"]), (10, ["B"], ["C"]), (20, ["B"], ["D"])])
    wt = main.build_wallet_transfer_graph(df)
    assert main.linear_chain_length(wt, "B") == 0


def test_min_receive_to_forward_minutes_measures_hold_time():
    # B receives at t=0, forwards at t=3 min
    df = _df([(0, ["A"], ["B"]), (3, ["B"], ["C"])])
    assert main.min_receive_to_forward_minutes(df, "B") == pytest.approx(3.0)


def test_min_receive_to_forward_minutes_capped_when_never_forwarded():
    df = _df([(0, ["A"], ["B"])])
    assert main.min_receive_to_forward_minutes(df, "B") == 1440.0


def test_features_appear_in_compute_wallet_features():
    df = _df([(0, ["A"], ["B"]), (3, ["B"], ["C"]), (6, ["C"], ["A"])])
    g = main.build_graph(df)
    feats = main.compute_wallet_features(df, g)
    for col in ("min_return_cycle_hops", "linear_chain_length", "min_receive_to_forward_minutes"):
        assert col in feats.columns
        assert col in main.FEATURE_COLS
