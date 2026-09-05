"""Tests for src/risk_model.py -- spec 3c (learned weights) and 3d (risk vs confidence)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import risk_model as rm  # noqa: E402

TX = str(ROOT / "data" / "synthetic_transactions.csv")
LABELS = str(ROOT / "data" / "synthetic_labels.csv")


@pytest.fixture(scope="module")
def scored():
    table, fit = rm.score_entities(TX, LABELS)
    return table, fit


def test_four_evidence_signals_present(scored):
    table, _ = scored
    for col in ("ai_evidence", "graph_evidence", "behavioral_evidence", "cluster_evidence"):
        assert col in table.columns


def test_cluster_evidence_is_a_zero_stub(scored):
    table, _ = scored
    assert (table["cluster_evidence"] == 0).all()


def test_learned_weights_differ_from_naive(scored):
    _, fit = scored
    w = fit["normalised_weights"]
    naive = {"ai": 0.4, "graph": 0.2, "behavioral": 0.2, "cluster": 0.2}
    # at least one weight moved by more than 0.1 -- fitting actually did something
    assert any(abs(w[k] - naive[k]) > 0.1 for k in naive)
    # cluster (a zero stub) must carry ~no weight
    assert w["cluster"] < 0.05


def test_fitting_changes_the_ranking(scored):
    table, _ = scored
    fit_rank = table["risk_fitted"].rank(ascending=False, method="min")
    naive_rank = table["risk_naive"].rank(ascending=False, method="min")
    assert (fit_rank != naive_rank).any()


def test_risk_and_confidence_are_separate_and_can_disagree(scored):
    table, _ = scored
    assert "risk_fitted" in table.columns
    assert "confidence" in table.columns
    # they are not the same number rescaled: correlation is far from perfect
    corr = table["risk_fitted"].corr(table["confidence"] * 100)
    assert abs(corr) < 0.95
    # at least one entity is high-risk but not high-confidence
    assert ((table["risk_fitted"] > 60) & (table["confidence"] < 0.8)).any()


def test_risk_is_bucketed_on_the_prototype_thresholds(scored):
    table, _ = scored
    assert set(table["risk_bucket"]) <= {"Low", "Medium", "High", "Critical"}
    for _, r in table.iterrows():
        if r["risk_fitted"] <= 30:
            assert r["risk_bucket"] == "Low"
        elif r["risk_fitted"] > 80:
            assert r["risk_bucket"] == "Critical"


def test_anomalous_entities_score_higher_on_average(scored):
    table, _ = scored
    anom = table[table["label"] == "anomalous"]["risk_fitted"].mean()
    norm = table[table["label"] == "normal"]["risk_fitted"].mean()
    assert anom > norm
