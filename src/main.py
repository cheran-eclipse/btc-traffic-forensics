"""
SIH26146 -- AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
Prototype pipeline: ingest -> correlate (network layer + blockchain layer) ->
graph -> AI/ML anomaly detection -> ranked, explainable leads -> visualization.

Run:
    python src/main.py --input data/sample_transactions.csv --top 10

This runs end-to-end on the bundled synthetic sample so the pipeline is
provably working before the real dataset (from the official PS Google Drive
link) is dropped in to replace data/sample_transactions.csv.
"""

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Ingestion
# ---------------------------------------------------------------------------

def load_transactions(path: str) -> pd.DataFrame:
    """Load the bulk metadata CSV described in the PS schema.

    Expected columns:
    timestamp, src_ip, dst_ip, src_port, dst_port, txid,
    input_addresses, output_addresses, input_amounts, output_amounts,
    geo_country, asn

    input_addresses / output_addresses / input_amounts / output_amounts are
    semicolon-separated when a transaction has multiple inputs or outputs
    (fan-in / fan-out), matching how a real Bitcoin tx is structured.
    """
    df = pd.read_csv(path, dtype=str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    def split_list(cell):
        return [x.strip() for x in str(cell).split(";") if x.strip()]

    df["input_addresses"] = df["input_addresses"].apply(split_list)
    df["output_addresses"] = df["output_addresses"].apply(split_list)
    df["input_amounts"] = df["input_amounts"].apply(
        lambda c: [float(x) for x in split_list(c)]
    )
    df["output_amounts"] = df["output_amounts"].apply(
        lambda c: [float(x) for x in split_list(c)]
    )
    return df


# ---------------------------------------------------------------------------
# 2. Correlation graph: wallets + IPs + transactions in one entity graph
# ---------------------------------------------------------------------------

def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Build a heterogeneous graph linking IPs, wallets and transactions.

    Node types (stored as node attr 'kind'): 'ip', 'wallet', 'tx'
    Edge types: ip->tx (relayed), tx->wallet (input), wallet->tx (output... )
    Storing both directions lets us later compute fan-in/fan-out per wallet
    and per-IP transaction counts, and lets a link-analysis view render
    IP <-> wallet <-> IP paths for an investigator.
    """
    g = nx.MultiDiGraph()

    for _, row in df.iterrows():
        tx = row["txid"]
        g.add_node(tx, kind="tx", timestamp=row["timestamp"])

        for ip_col in ("src_ip", "dst_ip"):
            ip = row[ip_col]
            g.add_node(ip, kind="ip")
            g.add_edge(ip, tx, kind="network_link", geo=row["geo_country"], asn=row["asn"])

        for addr, amt in zip(row["input_addresses"], row["input_amounts"] or [None] * len(row["input_addresses"])):
            g.add_node(addr, kind="wallet")
            g.add_edge(addr, tx, kind="input", amount=amt)

        for addr, amt in zip(row["output_addresses"], row["output_amounts"] or [None] * len(row["output_addresses"])):
            g.add_node(addr, kind="wallet")
            g.add_edge(tx, addr, kind="output", amount=amt)

    return g


# ---------------------------------------------------------------------------
# 3. Feature extraction per wallet (this is what the AI/ML model sees)
# ---------------------------------------------------------------------------

def build_wallet_transfer_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Wallet -> wallet projection: an edge a->b for every transaction where a
    was an input and b was an output, carrying the list of transfer timestamps.

    The main graph (build_graph) routes every wallet-to-wallet flow *through* a
    tx node, so the per-wallet features below only ever see a wallet's immediate
    neighbours. Multi-hop questions -- "do these funds come back to where they
    started?", "how long is the chain of pass-through wallets?" -- need this
    flattened view where wallet reachability is a direct path.
    """
    wt = nx.DiGraph()
    for _, row in df.iterrows():
        ts = row["timestamp"]
        for a in row["input_addresses"]:
            for b in row["output_addresses"]:
                if a == b:
                    continue
                if wt.has_edge(a, b):
                    wt[a][b]["timestamps"].append(ts)
                else:
                    wt.add_edge(a, b, timestamps=[ts])
    return wt


def min_return_cycle_hops(wt: nx.DiGraph, wallet: str, max_hops: int = 12) -> int:
    """Fewest wallet-hops for value leaving `wallet` to arrive back at `wallet`;
    0 if it never returns within `max_hops`.

    Why the old features can't see this: fan_in / fan_out / num_tx_touched are
    all counts of a wallet's *own* edges. A circular flow (A->B->C->A) is a
    property of a directed path several hops away, not of any single wallet's
    degree -- every wallet in the ring looks locally ordinary.
    """
    if wallet not in wt:
        return 0
    frontier = {wallet}
    seen = {wallet}
    for hop in range(1, max_hops + 1):
        nxt = set()
        for u in frontier:
            for v in wt.successors(u):
                if v == wallet:
                    return hop
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        if not nxt:
            break
        frontier = nxt
    return 0


def linear_chain_length(wt: nx.DiGraph, wallet: str, max_len: int = 40) -> int:
    """Length (in wallets) of the maximal 'pass-through' chain running through
    `wallet` -- the run of consecutive wallets each of which has at most one
    inbound and one outbound wallet counterpart in the whole dataset.

    Why the old features can't see this: layering is a *long* obfuscation chain
    A->B->C->...->J. Each hop wallet has fan_in == fan_out == 1, which reads as
    perfectly normal locally. The chain is only visible by walking it; depth is
    not any single wallet's attribute.
    """
    if wallet not in wt:
        return 0

    def chainable(n: str) -> bool:
        return wt.in_degree(n) <= 1 and wt.out_degree(n) <= 1

    if not chainable(wallet):
        return 0

    length = 1
    # walk backward
    cur = wallet
    steps = 0
    while steps < max_len:
        preds = list(wt.predecessors(cur))
        if len(preds) != 1 or not chainable(preds[0]) or preds[0] == wallet:
            break
        cur = preds[0]
        length += 1
        steps += 1
    # walk forward
    cur = wallet
    steps = 0
    while steps < max_len:
        succs = list(wt.successors(cur))
        if len(succs) != 1 or not chainable(succs[0]) or succs[0] == wallet:
            break
        cur = succs[0]
        length += 1
        steps += 1
    return length


def min_receive_to_forward_minutes(df: pd.DataFrame, wallet: str) -> float:
    """Smallest gap between `wallet` receiving value and then sending value on.
    Capped at 1440 (24 h); also returns 1440 if the wallet never both receives
    and later sends. Above a day the exact hold time doesn't matter -- it just
    isn't "rapid" -- and an uncapped sentinel would wreck feature scaling.

    Why the old features can't see this: time_span_minutes is the span between a
    wallet's first and last activity. Rapid movement is about *hold time* -- a
    mule wallet that receives and forwards within a minute -- which a total-span
    figure actively hides (it looks the same as a wallet used twice a month
    apart).
    """
    received, sent = [], []
    for _, row in df.iterrows():
        if wallet in row["output_addresses"]:
            received.append(row["timestamp"])
        if wallet in row["input_addresses"]:
            sent.append(row["timestamp"])
    best = None
    for r in received:
        for s in sent:
            if s >= r:
                gap = (s - r).total_seconds() / 60.0
                best = gap if best is None else min(best, gap)
    return min(float(best), 1440.0) if best is not None else 1440.0


def compute_wallet_features(df: pd.DataFrame, g: nx.MultiDiGraph) -> pd.DataFrame:
    """
    NOTE on fan_out: a wallet's own out-degree in this graph only counts how
    many transactions it was an *input* to -- it does NOT capture a classic
    peeling-chain (one input wallet splitting into many new output wallets
    in a single tx), because that split shows up as the *transaction's*
    output count, not an edge leaving the input wallet directly. That
    pattern is captured below as `max_tx_output_fanout` instead. Caught this
    by testing against a synthetic peeling-chain row before trusting it --
    same discipline as the rest of this project: verify against a known
    case, don't assume the first version of a feature is measuring what its
    name says it measures.

    Night 2: three multi-hop features were added (min_return_cycle_hops,
    linear_chain_length, min_receive_to_forward_minutes). They are new columns;
    the existing columns are unchanged. FEATURE_COLS was extended to include
    them, so the Isolation Forest now sees an 8-dimensional feature vector
    instead of 6 -- this is deliberate: the Night-1 diagnostic showed
    circular_flow, layering and rapid_movement were nearly invisible to the
    original six local features.
    """
    rows = []
    wallets = [n for n, d in g.nodes(data=True) if d.get("kind") == "wallet"]
    wt = build_wallet_transfer_graph(df)

    for w in wallets:
        fan_in = g.in_degree(w)   # times this wallet received (was an output)
        fan_out = g.out_degree(w)  # times this wallet spent (was an input)

        # transactions this wallet touches, to pull geo/timing/value context
        touching_tx = list(g.successors(w)) + list(g.predecessors(w))
        touching_tx = [t for t in touching_tx if g.nodes[t].get("kind") == "tx"]

        # peeling-chain signal: for each tx this wallet fed as an INPUT,
        # how many output wallets did that tx split into?
        max_tx_output_fanout = 0
        for tx in g.successors(w):
            if g.nodes[tx].get("kind") == "tx":
                n_outputs = sum(
                    1 for _, _, e in g.out_edges(tx, data=True) if e.get("kind") == "output"
                )
                max_tx_output_fanout = max(max_tx_output_fanout, n_outputs)

        countries, timestamps, amounts = set(), [], []
        for tx in touching_tx:
            for ip in list(g.predecessors(tx)):
                if g.nodes[ip].get("kind") == "ip":
                    edge_data = g.get_edge_data(ip, tx)
                    for e in edge_data.values():
                        if "geo" in e:
                            countries.add(e["geo"])
            ts = g.nodes[tx].get("timestamp")
            if ts is not None:
                timestamps.append(ts)

        for _, amt, kind in [
            (None, e.get("amount"), e.get("kind"))
            for _, _, e in g.edges(w, data=True)
        ]:
            if amt is not None:
                amounts.append(amt)

        time_span_minutes = 0.0
        if len(timestamps) >= 2:
            time_span_minutes = (max(timestamps) - min(timestamps)).total_seconds() / 60.0

        rows.append(
            {
                "wallet": w,
                "fan_in": fan_in,
                "fan_out": fan_out,
                "distinct_countries": len(countries),
                "countries": sorted(countries),
                "num_tx_touched": len(set(touching_tx)),
                "time_span_minutes": time_span_minutes,
                "avg_amount": float(np.mean(amounts)) if amounts else 0.0,
                "max_tx_output_fanout": max_tx_output_fanout,
                # -- Night 2 multi-hop features --
                "min_return_cycle_hops": min_return_cycle_hops(wt, w),
                "linear_chain_length": linear_chain_length(wt, w),
                "min_receive_to_forward_minutes": min_receive_to_forward_minutes(df, w),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. AI/ML anomaly detection (Isolation Forest) + rule-based explainability
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "fan_in", "fan_out", "distinct_countries", "num_tx_touched",
    "time_span_minutes", "max_tx_output_fanout",
    # Night 2 multi-hop features (see compute_wallet_features)
    "min_return_cycle_hops", "linear_chain_length", "min_receive_to_forward_minutes",
]


def flag_anomalies(features: pd.DataFrame) -> pd.DataFrame:
    X = features[FEATURE_COLS].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )
    model.fit(X_scaled)

    features = features.copy()
    features["anomaly_score"] = -model.decision_function(X_scaled)  # higher = more anomalous
    features["is_flagged"] = model.predict(X_scaled) == -1

    features["reason"] = features.apply(_explain_row, axis=1)
    return features.sort_values("anomaly_score", ascending=False)


def _explain_row(row) -> str:
    """Turn the raw features into a plain-language reason an investigator
    can read directly -- this is the 'explainable' half of the requirement,
    not just a bare confidence number."""
    reasons = []
    if row["max_tx_output_fanout"] >= 8:
        reasons.append(f"fed a transaction that split into {int(row['max_tx_output_fanout'])} outputs -- peeling-chain pattern")
    if row["distinct_countries"] >= 3:
        reasons.append(f"funds touched {int(row['distinct_countries'])} different countries")
    if row["time_span_minutes"] > 0 and row["time_span_minutes"] < 10 and row["num_tx_touched"] >= 3:
        reasons.append(f"{int(row['num_tx_touched'])} transactions within {row['time_span_minutes']:.1f} min (rapid layering)")
    if row.get("min_return_cycle_hops", 0) and row["min_return_cycle_hops"] > 0:
        reasons.append(f"funds return to this wallet after {int(row['min_return_cycle_hops'])} hops (circular flow)")
    if row.get("linear_chain_length", 0) and row["linear_chain_length"] >= 5:
        reasons.append(f"sits on a {int(row['linear_chain_length'])}-wallet pass-through chain (layering)")
    if row.get("min_receive_to_forward_minutes", 1440.0) < 5.0:
        reasons.append(f"forwarded funds {row['min_receive_to_forward_minutes']:.1f} min after receiving them (rapid movement)")
    if not reasons:
        reasons.append("statistically unusual combination of fan-in/fan-out/geo features")
    return "; ".join(reasons)


# ---------------------------------------------------------------------------
# 5. Visualization (baseline link-analysis view; swap for a real dashboard later)
# ---------------------------------------------------------------------------

def save_graph_snapshot(g: nx.MultiDiGraph, flagged_wallets: set, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_map = []
    for n, d in g.nodes(data=True):
        if n in flagged_wallets:
            color_map.append("#e04a4a")
        elif d.get("kind") == "wallet":
            color_map.append("#4a86e8")
        elif d.get("kind") == "ip":
            color_map.append("#43d692")
        else:
            color_map.append("#cccccc")

    plt.figure(figsize=(11, 8))
    pos = nx.spring_layout(g, seed=42, k=0.6)
    nx.draw(
        g, pos, node_color=color_map, node_size=180, with_labels=False,
        arrows=True, arrowsize=6, width=0.5, edge_color="#999999",
    )
    plt.title("Bitcoin transaction traffic: entity graph (red = flagged)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/sample_transactions.csv")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out-csv", default="output/ranked_leads.csv")
    parser.add_argument("--out-png", default="output/entity_graph.png")
    args = parser.parse_args()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading transactions from {args.input}")
    df = load_transactions(args.input)
    print(f"      {len(df)} transactions loaded")

    print("[2/5] Building IP <-> wallet <-> tx correlation graph")
    g = build_graph(df)
    print(f"      {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    print("[3/5] Extracting per-wallet features")
    features = compute_wallet_features(df, g)

    print("[4/5] Running anomaly model + generating explanations")
    ranked = flag_anomalies(features)

    print(f"[5/5] Writing top {args.top} leads to {args.out_csv} and graph to {args.out_png}")
    ranked.head(args.top).drop(columns=["countries"]).to_csv(args.out_csv, index=False)

    flagged = set(ranked[ranked["is_flagged"]]["wallet"])
    save_graph_snapshot(g, flagged, args.out_png)

    print("\n=== TOP RANKED INVESTIGATIVE LEADS ===")
    for _, row in ranked.head(args.top).iterrows():
        flag_marker = "FLAGGED" if row["is_flagged"] else "watch"
        print(f"[{flag_marker}] {row['wallet'][:20]:22s} score={row['anomaly_score']:.3f}  {row['reason']}")


if __name__ == "__main__":
    main()
