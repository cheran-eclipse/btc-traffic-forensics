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
    """
    rows = []
    wallets = [n for n, d in g.nodes(data=True) if d.get("kind") == "wallet"]

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
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. AI/ML anomaly detection (Isolation Forest) + rule-based explainability
# ---------------------------------------------------------------------------

FEATURE_COLS = ["fan_in", "fan_out", "distinct_countries", "num_tx_touched", "time_span_minutes", "max_tx_output_fanout"]


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
