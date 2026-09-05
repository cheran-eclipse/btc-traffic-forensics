"""Known-answer tests for src/generate_dataset.py (spec section 3b).

The generator controls the ground truth, so every property below is true by
construction -- these tests fail only if the generator stops planting what its
labels claim.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_dataset import ANOMALY_TYPES, BITCOIN_P2P_PORT, build_dataset  # noqa: E402


@pytest.fixture(scope="module")
def dataset():
    return build_dataset(seed=7, n_normal=45, instances_per_anomaly=3)


def _label_of(labels):
    return {row["entity"]: (row["label"], row["anomaly_type"]) for row in labels}


def test_enough_transactions_for_training(dataset):
    assert dataset["summary"]["n_transactions"] >= 200


def test_each_anomaly_type_has_the_requested_instances(dataset):
    injected = dataset["summary"]["injected_instances"]
    for atype in ANOMALY_TYPES:
        assert injected.get(atype) == 3, f"{atype}: {injected.get(atype)}"
    # every planted instance also labels at least one entity
    for atype in ANOMALY_TYPES:
        assert dataset["summary"]["by_anomaly_type"][atype] >= 3


def test_normal_and_anomalous_split_is_reported(dataset):
    s = dataset["summary"]
    assert s["n_normal"] > s["n_anomalous"] > 0
    assert s["n_normal"] + s["n_anomalous"] == s["n_entities"]


def test_every_wallet_in_the_data_is_labelled(dataset):
    labelled = {row["entity"] for row in dataset["labels"]}
    seen = set()
    for tx in dataset["transactions"]:
        seen.update(tx["input_addresses"])
        seen.update(tx["output_addresses"])
    assert seen == labelled


def test_peeling_chain_subjects_really_feed_a_high_fanout_tx(dataset):
    instances = [i for i in dataset["instances"] if i["anomaly_type"] == "peeling_chain"]
    assert len(instances) == 3
    labels = _label_of(dataset["labels"])
    for inst in instances:
        for w in inst["members"]:
            assert labels[w] == ("anomalous", "peeling_chain")
            fed = [tx for tx in dataset["transactions"] if w in tx["input_addresses"]]
            assert any(len(tx["output_addresses"]) >= 8 for tx in fed), w


def test_circular_flow_returns_funds_to_the_origin(dataset):
    instances = [i for i in dataset["instances"] if i["anomaly_type"] == "circular_flow"]
    assert len(instances) == 3
    for inst in instances:
        origin = inst["origin"]
        spent_at = [tx["timestamp"] for tx in dataset["transactions"] if origin in tx["input_addresses"]]
        received_at = [tx["timestamp"] for tx in dataset["transactions"] if origin in tx["output_addresses"]]
        assert spent_at and received_at
        assert max(received_at) > min(spent_at)  # funds came back after leaving


def test_normal_wallets_never_feed_a_peeling_fanout(dataset):
    labels = _label_of(dataset["labels"])
    normal = {w for w, (lbl, _) in labels.items() if lbl == "normal"}
    for tx in dataset["transactions"]:
        if len(tx["output_addresses"]) >= 8:
            assert not (set(tx["input_addresses"]) & normal)


def test_schema_amounts_line_up_with_addresses(dataset):
    for tx in dataset["transactions"]:
        assert len(tx["input_addresses"]) == len(tx["input_amounts"])
        assert len(tx["output_addresses"]) == len(tx["output_amounts"])
        assert tx["dst_port"] == BITCOIN_P2P_PORT
        assert tx["txid"].startswith("tx")


def test_generation_is_deterministic_for_a_seed():
    a = build_dataset(seed=7)
    b = build_dataset(seed=7)
    assert a["summary"] == b["summary"]
    assert a["labels"] == b["labels"]
    assert [
        (t["txid"], t["input_addresses"], t["output_addresses"]) for t in a["transactions"]
    ] == [
        (t["txid"], t["input_addresses"], t["output_addresses"]) for t in b["transactions"]
    ]


def test_a_different_seed_changes_the_data():
    a = build_dataset(seed=7)
    b = build_dataset(seed=8)
    assert a["labels"] != b["labels"]


def test_transactions_are_time_ordered(dataset):
    ts = [tx["timestamp"] for tx in dataset["transactions"]]
    assert ts == sorted(ts)
