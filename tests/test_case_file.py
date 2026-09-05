"""Tests for src/case_file.py -- spec 3e investigative lead format."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import case_file  # noqa: E402
from generate_dataset import build_dataset  # noqa: E402

TX = str(ROOT / "data" / "synthetic_transactions.csv")
LABELS = str(ROOT / "data" / "synthetic_labels.csv")

_DS = build_dataset(seed=7)
PEELING = [i["origin"] for i in _DS["instances"] if i["anomaly_type"] == "peeling_chain"]
CIRCULAR = [i["origin"] for i in _DS["instances"] if i["anomaly_type"] == "circular_flow"]


@pytest.fixture(scope="module")
def peeling_files():
    return case_file.generate(TX, LABELS, wallets=PEELING)


@pytest.fixture(scope="module")
def circular_files():
    return case_file.generate(TX, LABELS, wallets=CIRCULAR)


def test_all_3e_fields_present(peeling_files):
    assert len(peeling_files) == 3
    for cf in peeling_files:
        for key in ("entity", "risk_score", "confidence_score", "why_flagged",
                    "supporting_txids", "related_entities", "timeline",
                    "investigation_path"):
            assert key in cf


def test_risk_and_confidence_are_separate_numbers(peeling_files):
    for cf in peeling_files:
        assert isinstance(cf["risk_score"], float)
        assert isinstance(cf["confidence_score"], float)
        assert 0 <= cf["risk_score"] <= 100
        assert 0 <= cf["confidence_score"] <= 1
        # not the same figure on two scales
        assert abs(cf["risk_score"] / 100 - cf["confidence_score"]) > 0.05


def test_peeling_case_file_names_the_fan_out_pattern(peeling_files):
    for cf in peeling_files:
        joined = " ".join(cf["why_flagged"]).lower()
        assert "fan_out" in joined and "outputs" in joined
        assert cf["supporting_txids"]


def test_circular_case_file_names_the_cycle_and_traces_it_back(circular_files):
    for cf in circular_files:
        joined = " ".join(cf["why_flagged"]).lower()
        assert "circular_flow" in joined
        path = cf["investigation_path"]
        assert path.startswith("IP ")
        assert "TXID " in path and path.count("Wallet ") >= 2


def test_timeline_is_time_ordered(circular_files):
    for cf in circular_files:
        stamps = [line[:16] for line in cf["timeline"]]
        assert stamps == sorted(stamps)


def test_no_overclaiming_language(peeling_files, circular_files):
    banned = ["proves", "criminal", "identifies the owner", "proof of ownership", "guilty"]
    for cf in peeling_files + circular_files:
        text = case_file.format_case_file(cf).lower()
        for b in banned:
            assert b not in text


def test_format_is_printable(peeling_files):
    s = case_file.format_case_file(peeling_files[0])
    assert "INVESTIGATIVE LEAD" in s and "INVESTIGATION PATH" in s
