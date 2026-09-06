"""
SIH26146 / BitGuard AI -- section 3e: the Investigative Lead case file.

Replaces the flat `wallet, score, reason` print with a structured record per
flagged entity:

    Entity, Risk score, Confidence score,
    Why flagged        -- specific reasons
    Supporting evidence -- specific TXIDs
    Related entities    -- wallets / IPs that share a transaction
    Timeline            -- the entity's transactions in time order
    Investigation path  -- IP -> TXID -> Wallet -> Wallet ...
    Subgraph            -- a compact spec of just this lead's money path,
                          rendered by subgraph.render (dashboard "money path")

Risk and Confidence are shown as two separate numbers (spec 3d). Every reason
is phrased as behaviour that warrants investigation -- never "identifies a
criminal" or "proves ownership" (spec section 4).

Offline: standard library + the already-built pipeline objects.

Run:
    python src/case_file.py --tx data/synthetic_transactions.csv \
                            --labels data/synthetic_labels.csv --top 3
    python src/case_file.py ... --wallets 1QSJm5...,1fAJSe...
"""

from __future__ import annotations

import argparse
import textwrap

import networkx as nx
import pandas as pd

import main as pipeline
import pattern_heuristics as ph
import risk_model
import subgraph


def _tx_time(g, tx):
    return g.nodes[tx].get("timestamp")


def _tx_wallets(g, tx, kind):
    if kind == "input":
        return [u for u, _, e in g.in_edges(tx, data=True) if e.get("kind") == "input"]
    return [v for _, v, e in g.out_edges(tx, data=True) if e.get("kind") == "output"]


def _tx_ips(g, tx):
    return sorted({u for u, _, e in g.in_edges(tx, data=True)
                   if e.get("kind") == "network_link"})


def _touched_txs(g, wallet):
    if wallet not in g:
        return []
    nbrs = list(g.successors(wallet)) + list(g.predecessors(wallet))
    txs = {n for n in nbrs if g.nodes[n].get("kind") == "tx"}
    return sorted(txs, key=lambda t: _tx_time(g, t))


def _why_flagged(ranked_row, fired: list[ph.PatternSignal]) -> list[str]:
    """Named pattern signals are the source of truth. The Isolation Forest's
    free-text reason only contributes what the six heuristics don't cover
    (the geographic-spread signal), so nothing is stated twice."""
    reasons = [f"[{s.name}] {s.detail}" for s in fired]

    if int(ranked_row.get("distinct_countries", 0)) >= 3:
        reasons.append(
            f"[geo_spread] funds touched {int(ranked_row['distinct_countries'])} "
            f"different countries"
        )

    if not reasons:
        reasons.append(
            "statistically unusual combination of features vs. the rest of the "
            "dataset (no single named pattern) -- warrants a look, not a conclusion"
        )
    return reasons


def _timeline(g, wallet, df) -> list[str]:
    events = []
    for tx in _touched_txs(g, wallet):
        ts = _tx_time(g, tx)
        ins, outs = _tx_wallets(g, tx, "input"), _tx_wallets(g, tx, "output")
        role = "sent" if wallet in ins else "received"
        other = outs if role == "sent" else ins
        other = [w for w in other if w != wallet]
        shown = ", ".join(w[:12] for w in other[:4]) + ("  ..." if len(other) > 4 else "")
        events.append(
            f"{ts:%Y-%m-%d %H:%M}  {role:8s} via {tx}  "
            f"({len(ins)} in / {len(outs)} out)  -> {shown}"
        )
    return events


def _related_entities(g, wallet) -> dict:
    wallets, ips = set(), set()
    for tx in _touched_txs(g, wallet):
        wallets.update(_tx_wallets(g, tx, "input"))
        wallets.update(_tx_wallets(g, tx, "output"))
        ips.update(_tx_ips(g, tx))
    wallets.discard(wallet)
    return {"wallets": sorted(wallets), "ips": sorted(ips)}


def _investigation_path(g, wt, wallet) -> str:
    """IP -> TXID -> Wallet -> Wallet chain, following the money forward."""
    txs = _touched_txs(g, wallet)
    if not txs:
        return f"(no linked transactions) {wallet}"
    # prefer a tx the wallet fed (money leaving), else the first
    fed = [t for t in txs if wallet in _tx_wallets(g, t, "input")]
    tx = fed[0] if fed else txs[0]
    ips = _tx_ips(g, tx)
    ip = ips[0] if ips else "?"

    hops = [f"IP {ip}", f"TXID {tx}", f"Wallet {wallet}"]
    cur, seen = wallet, {wallet}
    for _ in range(4):
        if cur not in wt:
            break
        nxts = [v for v in wt.successors(cur) if v not in seen]
        if not nxts:
            break
        cur = nxts[0]
        seen.add(cur)
        hops.append(f"Wallet {cur}")
    return "  ->  ".join(hops)


def build_case_file(wallet, df, g, wt, ranked_row, risk_row) -> dict:
    feat = ranked_row.to_dict()
    fired = ph.fired_signals(g, wt, df, wallet, feat)

    evidence_txids = sorted({t for s in fired for t in s.txids}
                            or set(_touched_txs(g, wallet)))
    related = _related_entities(g, wallet)

    return {
        "entity": wallet,
        # per-lead subgraph spec (reuses g/wt + the flagged txids; renders via
        # subgraph.render). Replaces the old full-dataset hairball.
        "subgraph": subgraph.build_spec(g, wt, wallet, evidence_txids),
        "risk_score": float(risk_row["risk_fitted"]),
        "risk_bucket": str(risk_row["risk_bucket"]),
        "confidence_score": float(risk_row["confidence"]),
        "confidence_breakdown": {
            "correlation_confidence": float(risk_row["corr_confidence"]),
            "feature_completeness": float(risk_row["feature_completeness"]),
        },
        "ground_truth_label": risk_row.get("anomaly_type", "unknown"),
        "why_flagged": _why_flagged(ranked_row, fired),
        "supporting_txids": evidence_txids,
        "related_entities": related,
        "timeline": _timeline(g, wallet, df),
        "investigation_path": _investigation_path(g, wt, wallet),
    }


def format_case_file(cf: dict) -> str:
    L = []
    L.append("=" * 72)
    L.append(f"INVESTIGATIVE LEAD -- {cf['entity']}")
    L.append("=" * 72)
    L.append(f"Risk score        : {cf['risk_score']:.1f} / 100   ({cf['risk_bucket']})")
    L.append(f"Confidence score  : {cf['confidence_score']:.2f}        "
             f"(corr {cf['confidence_breakdown']['correlation_confidence']:.2f} / "
             f"completeness {cf['confidence_breakdown']['feature_completeness']:.2f})")
    L.append("  Risk = how unusual the behaviour is.  Confidence = how much to "
             "trust the evidence.")
    L.append("  These are separate numbers and may disagree.")
    L.append(f"(synthetic ground-truth label: {cf['ground_truth_label']})")
    L.append("")
    L.append("WHY FLAGGED (behaviour warranting investigation -- not a conclusion):")
    for r in cf["why_flagged"]:
        L.append(f"  - {r}")
    L.append("")
    L.append(f"SUPPORTING EVIDENCE -- transactions ({len(cf['supporting_txids'])}):")
    L.append("  " + ", ".join(cf["supporting_txids"]) or "  (none)")
    L.append("")
    rel = cf["related_entities"]
    L.append(f"RELATED ENTITIES -- {len(rel['wallets'])} wallets, {len(rel['ips'])} IPs:")
    L.append("  wallets: " + (", ".join(w[:14] for w in rel["wallets"][:10])
                              + (" ..." if len(rel["wallets"]) > 10 else "")))
    L.append("  IPs    : " + ", ".join(rel["ips"][:10]))
    L.append("")
    L.append("TIMELINE:")
    for e in cf["timeline"]:
        L.append(f"  {e}")
    L.append("")
    L.append("INVESTIGATION PATH:")
    for line in textwrap.wrap(cf["investigation_path"], 96,
                              subsequent_indent="      "):
        L.append("  " + line)
    L.append("=" * 72)
    return "\n".join(L)


def generate(tx_csv, labels_csv, wallets=None, top=3):
    table, _ = risk_model.score_entities(tx_csv, labels_csv)
    df = pipeline.load_transactions(tx_csv)
    g = pipeline.build_graph(df)
    wt = pipeline.build_wallet_transfer_graph(df)
    ranked = pipeline.flag_anomalies(pipeline.compute_wallet_features(df, g)).set_index("wallet")

    if wallets:
        picked = wallets
    else:
        picked = list(table.head(top)["wallet"])

    files = []
    for w in picked:
        if w not in ranked.index:
            continue
        risk_row = table[table["wallet"] == w].iloc[0]
        files.append(build_case_file(w, df, g, wt, ranked.loc[w], risk_row))
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tx", default="data/synthetic_transactions.csv")
    ap.add_argument("--labels", default="data/synthetic_labels.csv")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--wallets", default="", help="comma-separated wallet ids")
    args = ap.parse_args()

    wallets = [w.strip() for w in args.wallets.split(",") if w.strip()] or None
    for cf in generate(args.tx, args.labels, wallets=wallets, top=args.top):
        print(format_case_file(cf))
        print()


if __name__ == "__main__":
    main()
