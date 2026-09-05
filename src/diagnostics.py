"""
SIH26146 / BitGuard AI -- detection diagnostic.

Runs the current pipeline (build_graph -> compute_wallet_features ->
flag_anomalies) against a labelled dataset and reports, per anomaly type, what
fraction of that type's labelled-anomalous wallets actually get flagged
(is_flagged == True). This is "recall per pattern" -- it tells us which planted
behaviours the model can already see and which are invisible to the current
features.

Used before/after adding features so the improvement is measurable, not assumed.

Run:
    python src/diagnostics.py --tx data/synthetic_transactions.csv \
                              --labels data/synthetic_labels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import main as pipeline


def recall_by_anomaly_type(tx_csv: str, labels_csv: str) -> pd.DataFrame:
    """Return a table: anomaly_type, n_wallets, n_flagged, recall."""
    labels = pd.read_csv(labels_csv)

    df = pipeline.load_transactions(tx_csv)
    g = pipeline.build_graph(df)
    features = pipeline.compute_wallet_features(df, g)
    ranked = pipeline.flag_anomalies(features)

    merged = labels.merge(
        ranked[["wallet", "is_flagged", "anomaly_score"]],
        left_on="entity",
        right_on="wallet",
        how="left",
    )
    merged["is_flagged"] = merged["is_flagged"].fillna(False)

    anom = merged[merged["label"] == "anomalous"]
    rows = []
    for atype, grp in anom.groupby("anomaly_type"):
        rows.append(
            {
                "anomaly_type": atype,
                "n_wallets": len(grp),
                "n_flagged": int(grp["is_flagged"].sum()),
                "recall": round(grp["is_flagged"].mean(), 3),
            }
        )
    # overall normal false-positive rate, for context
    normal = merged[merged["label"] == "normal"]
    rows.append(
        {
            "anomaly_type": "(normal wallets: false-positive rate)",
            "n_wallets": len(normal),
            "n_flagged": int(normal["is_flagged"].sum()),
            "recall": round(normal["is_flagged"].mean(), 3),
        }
    )
    return pd.DataFrame(rows)


def print_table(table: pd.DataFrame, title: str = "recall by anomaly type") -> None:
    print(f"\n=== {title} ===")
    print(f"{'anomaly_type':45s} {'wallets':>8s} {'flagged':>8s} {'recall':>8s}")
    for _, r in table.iterrows():
        print(f"{r['anomaly_type']:45s} {r['n_wallets']:8d} {r['n_flagged']:8d} {r['recall']:8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tx", default="data/synthetic_transactions.csv")
    ap.add_argument("--labels", default="data/synthetic_labels.csv")
    args = ap.parse_args()

    table = recall_by_anomaly_type(args.tx, args.labels)
    print_table(table)


if __name__ == "__main__":
    main()
