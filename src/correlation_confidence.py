"""
SIH26146 / BitGuard AI -- Section 3a: correlation confidence scoring.

The existing pipeline (src/main.py :: build_graph) links every IP to every
transaction it touched with an unweighted edge -- there is no notion of "how
sure are we that this packet relates to this transaction". A raw
nearest-timestamp match silently picks one candidate when several cluster in
the same window and reports false certainty.

This module makes the confidence number real. For one network-layer
observation (a packet / flow seen on the wire) and a set of candidate
transactions it *could* correspond to, it computes:

    confidence = 0.5 * time_score + 0.3 * port_score + 0.2 * ambiguity_penalty

    time_score       higher when the observation and the candidate transaction
                     are closer together in time (exponential half-life decay)
    port_score       1.0 when the observed port is the Bitcoin P2P port,
                     0.0 otherwise
    ambiguity_penalty 1 / (candidates in the same time window) -- a match
                     surrounded by many equally-plausible candidates scores
                     lower. Despite the name, higher = less ambiguous = better;
                     the name is kept to match the design document.

Matches at or above the accept threshold are labelled an "observation"
(network activity time-correlated with a transaction) -- never proof of wallet
ownership. Matches below threshold are returned as UNRESOLVED and left for an
analyst instead of being forced into a guess.

Offline: pure standard library. No network, no third-party imports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

# The Bitcoin peer-to-peer port. Kept as a named constant so a dataset that
# uses a different P2P port can override it via score_correlation(...).
BITCOIN_P2P_PORT = 8333

# Weights are taken verbatim from the BitGuard AI design document (spec 3a).
# They sum to 1.0, so confidence is always in [0.0, 1.0].
W_TIME = 0.5
W_PORT = 0.3
W_AMBIGUITY = 0.2

DEFAULT_ACCEPT_THRESHOLD = 0.5
DEFAULT_TIME_HALFLIFE_S = 60.0      # time_score = 0.5 at 60 s apart
DEFAULT_AMBIGUITY_WINDOW_S = 600.0  # candidates within +/- 10 min count as competing


@dataclass
class CorrelationResult:
    """Outcome of scoring one observation against its candidate transactions."""

    status: str                     # "ACCEPTED" or "UNRESOLVED"
    label: str                      # human-readable; always an "observation", never ownership
    confidence: float               # 0.0 - 1.0
    matched_txid: str | None        # best candidate (even when UNRESOLVED), or None if no candidates
    time_score: float
    port_score: float
    ambiguity_penalty: float
    candidates_in_window: int
    dt_seconds: float | None        # time gap to the matched candidate

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _seconds_between(a: datetime, b: datetime) -> float:
    return abs((a - b).total_seconds())


def time_score(dt_seconds: float, halflife_s: float = DEFAULT_TIME_HALFLIFE_S) -> float:
    """1.0 at dt=0, 0.5 at dt=halflife, approaching 0 as the gap grows."""
    if halflife_s <= 0:
        raise ValueError("halflife_s must be positive")
    if dt_seconds < 0:
        raise ValueError("dt_seconds must be non-negative")
    return float(0.5 ** (dt_seconds / halflife_s))


def port_score(observed_port: Any, btc_p2p_port: int = BITCOIN_P2P_PORT) -> float:
    """1.0 when the observed port is the Bitcoin P2P port, else 0.0."""
    try:
        return 1.0 if int(observed_port) == int(btc_p2p_port) else 0.0
    except (TypeError, ValueError):
        return 0.0


def score_correlation(
    observation: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    btc_p2p_port: int = BITCOIN_P2P_PORT,
    time_halflife_s: float = DEFAULT_TIME_HALFLIFE_S,
    ambiguity_window_s: float = DEFAULT_AMBIGUITY_WINDOW_S,
    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
) -> CorrelationResult:
    """Score how confidently one network observation maps to a transaction.

    observation: mapping with
        'timestamp' -> datetime  (when the packet / flow was seen)
        'port'      -> int-like  (the observed port; src or dst, caller's choice)
    candidates: sequence of mappings, each with
        'txid'      -> str
        'timestamp' -> datetime

    Returns a CorrelationResult. An ACCEPTED result is explicitly an
    observation-level correlation, not a claim of wallet ownership.
    """
    obs_ts: datetime = observation["timestamp"]
    p_score = port_score(observation.get("port"), btc_p2p_port)

    cand_list = list(candidates)
    if not cand_list:
        return CorrelationResult(
            status="UNRESOLVED",
            label="UNRESOLVED: no candidate transactions to correlate against",
            confidence=0.0,
            matched_txid=None,
            time_score=0.0,
            port_score=p_score,
            ambiguity_penalty=0.0,
            candidates_in_window=0,
            dt_seconds=None,
        )

    # Ambiguity: how many candidates sit in the same time window as the
    # observation. More competitors -> lower ambiguity_penalty -> lower score.
    in_window = [
        c for c in cand_list
        if _seconds_between(c["timestamp"], obs_ts) <= ambiguity_window_s
    ]
    n_window = len(in_window)
    denom = n_window if n_window else len(cand_list)
    ambiguity_penalty = 1.0 / denom

    # port_score and ambiguity_penalty are identical across candidates, so the
    # best match is simply the one closest in time (highest time_score).
    best_conf = -1.0
    best_txid: str | None = None
    best_time_score = 0.0
    best_dt = 0.0
    for c in cand_list:
        dt = _seconds_between(c["timestamp"], obs_ts)
        t_score = time_score(dt, time_halflife_s)
        conf = W_TIME * t_score + W_PORT * p_score + W_AMBIGUITY * ambiguity_penalty
        if conf > best_conf:
            best_conf, best_txid, best_time_score, best_dt = conf, c["txid"], t_score, dt

    status = "ACCEPTED" if best_conf >= accept_threshold else "UNRESOLVED"
    if status == "ACCEPTED":
        label = (
            f"observation: network activity time-correlated with {best_txid} "
            f"(confidence {best_conf:.2f}) -- NOT proof of wallet ownership"
        )
    else:
        label = (
            f"UNRESOLVED: best candidate {best_txid} scores {best_conf:.2f} "
            f"(below accept threshold {accept_threshold:.2f}); "
            f"left for an analyst, not guessed"
        )

    return CorrelationResult(
        status=status,
        label=label,
        confidence=round(best_conf, 6),
        matched_txid=best_txid,
        time_score=round(best_time_score, 6),
        port_score=p_score,
        ambiguity_penalty=round(ambiguity_penalty, 6),
        candidates_in_window=n_window,
        dt_seconds=best_dt,
    )


# ---------------------------------------------------------------------------
# Small self-check: run `python src/correlation_confidence.py` to see the two
# reference cases from spec section 6 (one unambiguous, one crowded window).
# ---------------------------------------------------------------------------

def _demo() -> None:
    from datetime import timedelta

    t0 = datetime(2026, 9, 1, 2, 14, 0)

    print("Case A -- single unambiguous candidate, Bitcoin P2P port:")
    a = score_correlation(
        {"timestamp": t0, "port": BITCOIN_P2P_PORT},
        [{"txid": "tx0001", "timestamp": t0 + timedelta(seconds=8)}],
    )
    for k, v in a.as_dict().items():
        print(f"    {k:22s} {v}")

    print("\nCase B -- same nearest candidate, but 8 competing in the window:")
    b = score_correlation(
        {"timestamp": t0, "port": BITCOIN_P2P_PORT},
        [{"txid": "tx0001", "timestamp": t0 + timedelta(seconds=8)}]
        + [
            {"txid": f"tx{i:04d}", "timestamp": t0 + timedelta(seconds=20 + 12 * i)}
            for i in range(2, 9)
        ],
    )
    for k, v in b.as_dict().items():
        print(f"    {k:22s} {v}")

    print("\nCase C -- weak match: far in time, wrong (ephemeral) port, crowded:")
    c = score_correlation(
        {"timestamp": t0, "port": 51413},
        [
            {"txid": f"tx{i:04d}", "timestamp": t0 + timedelta(seconds=240 + 30 * i)}
            for i in range(6)
        ],
    )
    for k, v in c.as_dict().items():
        print(f"    {k:22s} {v}")


if __name__ == "__main__":
    _demo()
