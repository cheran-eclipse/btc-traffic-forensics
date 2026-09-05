"""
SIH26146 / BitGuard AI -- the six pattern-detection heuristics (spec section 5
Night 3; spec section 4).

fan-in, fan-out, rapid movement, amount splitting, layering, circular flow.

These are SUPPORTING EVIDENCE for the Isolation Forest's verdict and material
for the investigative-lead case file -- **not** a competing anomaly call. The
spec is explicit (section 4): "Don't present four outputs that might disagree."
So each heuristic returns a named, explainable `PatternSignal` (did it fire,
why, which transactions), and nothing here decides whether a wallet is flagged.

Four of the six already exist as Night 2 features / explanation strings
(fan-out/peeling, rapid movement, layering chain, circular flow); this module
reuses those values and just packages them as structured named signals with
transaction-level evidence. `fan_in` (consolidation) and `amount_splitting`
(structuring into near-equal parts) are new named checks.

Offline: standard library only (operates on an already-built graph).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

# Prototype thresholds -- not tuned against anything official.
FAN_IN_MIN_INPUTS = 5          # a tx consolidating >=5 sources
FAN_OUT_MIN_OUTPUTS = 8        # a tx splitting into >=8 destinations
SPLIT_MIN_PARTS = 3            # >=3 outputs...
SPLIT_MAX_REL_SPREAD = 0.06   # ...whose values are within 6% of each other
RAPID_MOVE_MAX_HOLD_MIN = 5.0  # forwarded within 5 minutes of receiving
LAYERING_MIN_CHAIN = 5         # pass-through chain of >=5 wallets


@dataclass
class PatternSignal:
    name: str
    fired: bool
    detail: str = ""
    txids: list[str] = field(default_factory=list)


def _txs(g: nx.MultiDiGraph, wallet: str, direction: str) -> list[str]:
    if wallet not in g:
        return []
    nodes = g.successors(wallet) if direction == "fed" else g.predecessors(wallet)
    return [n for n in nodes if g.nodes[n].get("kind") == "tx"]


def _tx_input_wallets(g: nx.MultiDiGraph, tx: str) -> list[str]:
    return [u for u, _, e in g.in_edges(tx, data=True) if e.get("kind") == "input"]


def _tx_output_amounts(g: nx.MultiDiGraph, tx: str) -> list[float]:
    return [e["amount"] for _, _, e in g.out_edges(tx, data=True)
            if e.get("kind") == "output" and e.get("amount") is not None]


# --- the six -------------------------------------------------------------

def check_fan_in(g, wt, df, wallet, feat) -> PatternSignal:
    """Consolidation: the wallet took part in a transaction that pulled many
    separate sources together."""
    worst_tx, worst_n = None, 0
    for tx in set(_txs(g, wallet, "fed") + _txs(g, wallet, "received")):
        n = len(_tx_input_wallets(g, tx))
        if n > worst_n:
            worst_tx, worst_n = tx, n
    fired = worst_n >= FAN_IN_MIN_INPUTS
    return PatternSignal(
        "fan_in", fired,
        f"transaction {worst_tx} consolidated {worst_n} input wallets" if fired else "",
        [worst_tx] if fired else [],
    )


def check_fan_out(g, wt, df, wallet, feat) -> PatternSignal:
    """Peeling / split: the wallet fed a transaction that fanned out into many
    destinations (Night 2 feature max_tx_output_fanout, packaged with evidence)."""
    hits = [tx for tx in _txs(g, wallet, "fed")
            if len(_tx_output_amounts(g, tx)) >= FAN_OUT_MIN_OUTPUTS]
    n = int(feat.get("max_tx_output_fanout", 0))
    fired = bool(hits)
    return PatternSignal(
        "fan_out", fired,
        f"fed a transaction that split into {n} outputs (peeling-chain shape)" if fired else "",
        hits,
    )


def check_rapid_movement(g, wt, df, wallet, feat) -> PatternSignal:
    hold = float(feat.get("min_receive_to_forward_minutes", 1440.0))
    fired = hold < RAPID_MOVE_MAX_HOLD_MIN
    return PatternSignal(
        "rapid_movement", fired,
        f"forwarded funds {hold:.1f} min after receiving them" if fired else "",
        sorted(set(_txs(g, wallet, "received") + _txs(g, wallet, "fed")))[:6] if fired else [],
    )


def check_amount_splitting(g, wt, df, wallet, feat) -> PatternSignal:
    """Structuring: the wallet fed a transaction that broke a sum into several
    near-equal parts (distinct from fan_out, which is about count not equality)."""
    for tx in _txs(g, wallet, "fed"):
        amts = _tx_output_amounts(g, tx)
        if len(amts) < SPLIT_MIN_PARTS:
            continue
        lo, hi = min(amts), max(amts)
        if lo > 0 and (hi - lo) / hi <= SPLIT_MAX_REL_SPREAD:
            return PatternSignal(
                "amount_splitting", True,
                f"transaction {tx} split ~{sum(amts):.4f} into {len(amts)} "
                f"near-equal parts of ~{sum(amts)/len(amts):.4f}",
                [tx],
            )
    return PatternSignal("amount_splitting", False)


def check_layering(g, wt, df, wallet, feat) -> PatternSignal:
    n = int(feat.get("linear_chain_length", 0))
    fired = n >= LAYERING_MIN_CHAIN
    return PatternSignal(
        "layering", fired,
        f"sits on a {n}-wallet pass-through chain (long obfuscation chain)" if fired else "",
        sorted(set(_txs(g, wallet, "received") + _txs(g, wallet, "fed")))[:6] if fired else [],
    )


def check_circular_flow(g, wt, df, wallet, feat) -> PatternSignal:
    hops = int(feat.get("min_return_cycle_hops", 0))
    fired = hops > 0
    return PatternSignal(
        "circular_flow", fired,
        f"funds leaving this wallet return to it after {hops} hops" if fired else "",
        sorted(set(_txs(g, wallet, "fed") + _txs(g, wallet, "received")))[:6] if fired else [],
    )


_CHECKS = [
    check_fan_in, check_fan_out, check_rapid_movement,
    check_amount_splitting, check_layering, check_circular_flow,
]


def detect_all(g: nx.MultiDiGraph, wt: nx.DiGraph, df: pd.DataFrame,
               wallet: str, feat: dict) -> list[PatternSignal]:
    """All six signals for one wallet. `feat` is that wallet's row from
    compute_wallet_features (as a dict or Series)."""
    return [chk(g, wt, df, wallet, feat) for chk in _CHECKS]


def fired_signals(g, wt, df, wallet, feat) -> list[PatternSignal]:
    return [s for s in detect_all(g, wt, df, wallet, feat) if s.fired]


def _demo() -> None:
    import main as pipeline

    df = pipeline.load_transactions("data/synthetic_transactions.csv")
    g = pipeline.build_graph(df)
    wt = pipeline.build_wallet_transfer_graph(df)
    feats = pipeline.compute_wallet_features(df, g).set_index("wallet")

    from generate_dataset import build_dataset
    ds = build_dataset(seed=7)
    for atype in ("peeling_chain", "amount_splitting", "circular_flow"):
        w = next(i["origin"] for i in ds["instances"] if i["anomaly_type"] == atype)
        print(f"\n{atype}  {w}")
        for s in detect_all(g, wt, df, w, feats.loc[w].to_dict()):
            mark = "FIRED" if s.fired else "  -  "
            print(f"  [{mark}] {s.name:16s} {s.detail}")


if __name__ == "__main__":
    _demo()
