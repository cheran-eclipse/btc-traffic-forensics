"""Tests for src/subgraph.py -- the per-lead investigative subgraph.

Known-answer style: a peeling lead's subgraph must contain a fan-out
transaction; a circular-flow lead's must contain a ring-closing edge (one that
points backward along the flow). And the subgraph must be small -- a slice of
the case, not the whole dataset.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import case_file  # noqa: E402
import main as pipeline  # noqa: E402
import subgraph  # noqa: E402
from generate_dataset import build_dataset  # noqa: E402

TX = str(ROOT / "data" / "synthetic_transactions.csv")
LABELS = str(ROOT / "data" / "synthetic_labels.csv")
T0 = datetime(2026, 9, 1)

_DS = build_dataset(seed=7)
PEELING = next(i["origin"] for i in _DS["instances"] if i["anomaly_type"] == "peeling_chain")
CIRCULAR = next(i["origin"] for i in _DS["instances"] if i["anomaly_type"] == "circular_flow")


# -- hand-built known answers -------------------------------------------

def _df(rows):
    recs = []
    for i, (mins, ins, outs, in_amts, out_amts) in enumerate(rows):
        recs.append({
            "timestamp": T0 + timedelta(minutes=mins),
            "src_ip": "8.8.8.1", "dst_ip": "1.1.1.1", "src_port": 50000, "dst_port": 8333,
            "txid": f"tx{i:04d}", "input_addresses": ins, "output_addresses": outs,
            "input_amounts": in_amts, "output_amounts": out_amts,
            "geo_country": "US", "asn": "AS15169",
        })
    return pd.DataFrame(recs)


def _spec(df, entity, seed_txids):
    g = pipeline.build_graph(df)
    wt = pipeline.build_wallet_transfer_graph(df)
    return subgraph.build_spec(g, wt, entity, seed_txids)


def test_fan_out_tx_is_kept_as_a_node():
    outs = [f"o{i}" for i in range(11)]
    df = _df([(0, ["A"], outs, [2.0], [0.18] * 11)])
    spec = _spec(df, "A", ["tx0000"])
    tx = [n for n in spec["nodes"] if n["kind"] == "tx"]
    assert tx and tx[0]["n_out"] == 11
    # not every recipient is drawn -- the rest collapse to one node
    assert spec["collapsed"] > 0
    assert any(n["kind"] == "more" for n in spec["nodes"])


def test_ring_produces_a_backward_edge_in_the_layout():
    ring = list("ABCDE")
    rows = [(i * 10, [ring[i]], [ring[(i + 1) % 5]], [1.0], [0.98]) for i in range(5)]
    df = _df(rows)
    spec = _spec(df, "A", ["tx0000"])
    pos = subgraph.choose_layout(spec)
    money = [e for e in spec["edges"] if e["kind"] != "network_link"]
    backward = [e for e in money
                if e["src"] in pos and e["dst"] in pos
                and pos[e["dst"]][0] < pos[e["src"]][0] - 0.01]
    assert backward, "a ring must close with an edge pointing back along the flow"
    # the chain visits every ring wallet
    wallets = {n["id"] for n in spec["nodes"] if n["kind"] == "wallet"}
    assert set(ring) <= wallets


def test_simple_chain_collapses_transactions_onto_edges():
    df = _df([(0, ["A"], ["B"], [1.0], [0.99]),
              (10, ["B"], ["C"], [0.99], [0.98])])
    spec = _spec(df, "A", ["tx0000", "tx0001"])
    kinds = {e["kind"] for e in spec["edges"]}
    assert "transfer" in kinds          # 1-in/1-out tx became an edge
    assert not any(n["kind"] == "tx" for n in spec["nodes"])
    tr = next(e for e in spec["edges"] if e["kind"] == "transfer")
    assert tr["txid"] and tr["amount"] is not None


# -- against the real dataset -----------------------------------------

@pytest.fixture(scope="module")
def peeling_spec():
    return case_file.generate(TX, LABELS, wallets=[PEELING])[0]["subgraph"]


@pytest.fixture(scope="module")
def circular_spec():
    return case_file.generate(TX, LABELS, wallets=[CIRCULAR])[0]["subgraph"]


def test_case_file_now_carries_a_subgraph_spec(peeling_spec):
    for key in ("entity", "nodes", "edges", "n_tx", "n_wallets"):
        assert key in peeling_spec


def test_subgraph_is_a_slice_not_the_whole_dataset(peeling_spec, circular_spec):
    assert peeling_spec["n_wallets"] < 60      # dataset has 349 entities
    assert circular_spec["n_wallets"] < 20


def test_peeling_subgraph_shows_the_fan_out(peeling_spec):
    fan = [n for n in peeling_spec["nodes"] if n["kind"] == "tx" and n["n_out"] >= 8]
    assert fan, "peeling subgraph should contain a >=8-output transaction node"


def test_circular_subgraph_shows_the_ring_closing(circular_spec):
    pos = subgraph.choose_layout(circular_spec)
    money = [e for e in circular_spec["edges"] if e["kind"] != "network_link"]
    assert any(pos[e["dst"]][0] < pos[e["src"]][0] - 0.01
               for e in money if e["src"] in pos and e["dst"] in pos)


def test_the_two_patterns_render_differently(peeling_spec, circular_spec):
    p_fan = sum(1 for n in peeling_spec["nodes"] if n["kind"] == "tx" and n["n_out"] >= 8)
    c_fan = sum(1 for n in circular_spec["nodes"] if n["kind"] == "tx" and n["n_out"] >= 8)
    assert p_fan >= 1 and c_fan == 0        # peeling fans out, circular does not
    assert peeling_spec["collapsed"] > 0 and circular_spec["collapsed"] == 0


def test_key_nodes_have_text_labels(circular_spec):
    for n in circular_spec["nodes"]:
        if n["kind"] == "ip":
            assert any(ch.isdigit() for ch in n["label"])   # the IP address
        if n.get("role") in ("subject", "origin", "terminal"):
            assert len(n["label"]) > 5


def test_render_writes_a_png(tmp_path, circular_spec):
    out = subgraph.render(circular_spec, str(tmp_path / "sg.png"), "circular_flow")
    assert Path(out).is_file() and Path(out).stat().st_size > 2000
