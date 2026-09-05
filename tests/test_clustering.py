"""Tests for src/clustering.py (spec module 7, DBSCAN)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import clustering  # noqa: E402

TX = str(ROOT / "data" / "synthetic_transactions.csv")
LABELS = str(ROOT / "data" / "synthetic_labels.csv")


@pytest.fixture(scope="module")
def run():
    return clustering.run(TX, LABELS)


def test_produces_clusters_and_noise(run):
    clusters, report, _ = run
    assert clusters["cluster"].nunique() >= 3
    assert (clusters["cluster"] == -1).any()      # some noise
    assert (clusters["cluster"] >= 0).any()       # some real clusters


def test_anomalous_clusters_are_pure(run):
    _, report, _ = run
    real = report[report["cluster"] >= 0]
    pure_anom = real[real["frac_anomalous"] >= 0.8]
    # DBSCAN groups related bad actors, not noise: several pure clusters exist
    assert len(pure_anom) >= 3
    # and they carry a real chunk of the anomalous population
    assert pure_anom["n_anomalous"].sum() >= 30


def test_the_big_cluster_is_essentially_all_normal(run):
    _, report, _ = run
    biggest = report.loc[report["size"].idxmax()]
    assert biggest["frac_anomalous"] < 0.05


def test_cluster_evidence_is_label_free_and_nontrivial(run):
    clusters, _, features = run
    ev = clustering.cluster_evidence(features)
    assert set(ev.index) == set(features["wallet"])
    assert ev.nunique() > 1
    # distinct-group wallets score above mainstream/noise wallets
    distinct = clusters.set_index("wallet")["cluster"]
    mainstream = distinct[distinct >= 0].value_counts().idxmax()
    in_distinct = distinct[(distinct >= 0) & (distinct != mainstream)].index
    in_main = distinct[distinct == mainstream].index
    assert ev.reindex(in_distinct).mean() > ev.reindex(in_main).mean()


def test_deterministic(run):
    a = clustering.run(TX, LABELS)[0]
    b = clustering.run(TX, LABELS)[0]
    assert list(a["cluster"]) == list(b["cluster"])
