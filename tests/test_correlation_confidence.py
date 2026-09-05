"""Known-answer tests for src/correlation_confidence.py (spec section 3a).

Each test fixes an input where the right answer is known by construction, the
same discipline the README describes for catching the fan_out bug.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from correlation_confidence import (  # noqa: E402
    BITCOIN_P2P_PORT,
    W_AMBIGUITY,
    W_PORT,
    W_TIME,
    port_score,
    score_correlation,
    time_score,
)

T0 = datetime(2026, 9, 1, 2, 14, 0)


def _cand(txid, seconds):
    return {"txid": txid, "timestamp": T0 + timedelta(seconds=seconds)}


def test_time_score_halflife():
    assert time_score(0.0, halflife_s=60.0) == pytest.approx(1.0)
    assert time_score(60.0, halflife_s=60.0) == pytest.approx(0.5)
    assert time_score(120.0, halflife_s=60.0) == pytest.approx(0.25)


def test_port_score_is_binary():
    assert port_score(BITCOIN_P2P_PORT) == 1.0
    assert port_score(8333) == 1.0
    assert port_score(51413) == 0.0
    assert port_score(None) == 0.0
    assert port_score("not-a-port") == 0.0


def test_unambiguous_match_is_accepted_high():
    # One candidate, 8 s away, on the Bitcoin P2P port -> should score near 1.0.
    res = score_correlation(
        {"timestamp": T0, "port": BITCOIN_P2P_PORT},
        [_cand("tx0001", 8)],
    )
    assert res.status == "ACCEPTED"
    assert res.matched_txid == "tx0001"
    assert res.confidence > 0.9
    assert res.ambiguity_penalty == pytest.approx(1.0)  # 1 / 1 candidate


def test_crowded_window_scores_lower_than_unambiguous():
    obs = {"timestamp": T0, "port": BITCOIN_P2P_PORT}
    lone = score_correlation(obs, [_cand("tx0001", 8)])
    # Same nearest candidate, but 8 competitors in the same 10-min window.
    crowded = score_correlation(
        obs,
        [_cand("tx0001", 8)] + [_cand(f"tx{i:04d}", 20 + 12 * i) for i in range(2, 9)],
    )
    assert crowded.matched_txid == "tx0001"
    assert crowded.confidence < lone.confidence
    assert crowded.ambiguity_penalty == pytest.approx(1.0 / 8)
    assert crowded.candidates_in_window == 8


def test_weak_match_is_left_unresolved():
    # Far in time (>=4 min), ephemeral port, several equally-poor candidates.
    res = score_correlation(
        {"timestamp": T0, "port": 51413},
        [_cand(f"tx{i:04d}", 240 + 30 * i) for i in range(6)],
    )
    assert res.status == "UNRESOLVED"
    assert res.confidence < 0.5
    assert "not guessed" in res.label


def test_accepted_label_never_claims_ownership():
    res = score_correlation(
        {"timestamp": T0, "port": BITCOIN_P2P_PORT},
        [_cand("tx0001", 5)],
    )
    assert res.status == "ACCEPTED"
    lowered = res.label.lower()
    assert "observation" in lowered
    assert "ownership" not in lowered.replace("not proof of wallet ownership", "")
    assert "proves" not in lowered


def test_confidence_equals_the_spec_formula():
    res = score_correlation(
        {"timestamp": T0, "port": BITCOIN_P2P_PORT},
        [_cand("tx0001", 30), _cand("tx0002", 400)],
    )
    expected = (
        W_TIME * res.time_score
        + W_PORT * res.port_score
        + W_AMBIGUITY * res.ambiguity_penalty
    )
    assert res.confidence == pytest.approx(expected, abs=1e-6)


def test_no_candidates_is_unresolved():
    res = score_correlation({"timestamp": T0, "port": BITCOIN_P2P_PORT}, [])
    assert res.status == "UNRESOLVED"
    assert res.matched_txid is None
    assert res.confidence == 0.0
