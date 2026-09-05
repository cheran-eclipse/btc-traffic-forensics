"""
SIH26146 / BitGuard AI -- module 7: DBSCAN clustering of wallets.

Groups wallets by behavioural similarity over the SAME feature set the
Isolation Forest uses (main.FEATURE_COLS), scaled the same way (StandardScaler).
DBSCAN, not k-means: we don't know the number of groups, most wallets are
ordinary and should fall in one big blob or in noise, and the interesting
output is the small tight clusters.

This is SUPPORTING EVIDENCE, not a second verdict (spec section 4). Its job is
to answer "is this wallet grouped with other suspicious-looking wallets?" -- a
corroboration signal for the risk model and the case file, never an
independent anomaly call.

The purity report (fraction of each cluster that is labelled anomalous) is a
sanity check on the synthetic data: it tells us whether DBSCAN is actually
grouping related bad actors or just carving up noise.

cluster_evidence() is label-free (it uses the Isolation Forest anomaly score,
never the ground-truth labels) so it can feed the logistic regression in
risk_model without leakage.

Offline: numpy / pandas / scikit-learn only.

Run:
    python src/clustering.py --tx data/synthetic_transactions.csv \
                             --labels data/synthetic_labels.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import main as pipeline

# Prototype defaults. eps was chosen by sweeping 0.8..2.0 on the Night 1
# dataset: the five anomalous clusters (layering x2, circular_flow x2,
# peeling_chain) are stable across that whole range; 1.4 is where the noise
# set stops shrinking. Not tuned against anything official. The k-distance
# percentiles are printed for reference but did not drive the choice (the
# classic knee heuristic over-merges on this data).
DEFAULT_EPS = 1.4
DEFAULT_MIN_SAMPLES = 4


def _scaled_matrix(features: pd.DataFrame) -> np.ndarray:
    X = features[pipeline.FEATURE_COLS].fillna(0).to_numpy()
    return StandardScaler().fit_transform(X)


def _kdistance_percentiles(X: np.ndarray, min_samples: int) -> dict:
    """Percentiles of each point's distance to its kth neighbour (k =
    min_samples) -- context for choosing eps."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dists, _ = nn.kneighbors(X)
    kth = dists[:, -1]
    return {p: round(float(np.percentile(kth, p)), 2) for p in (50, 75, 90, 95)}


def cluster_wallets(
    features: pd.DataFrame,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> pd.DataFrame:
    """Return features + a 'cluster' column. Cluster -1 is DBSCAN noise."""
    X = _scaled_matrix(features)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    out = features[["wallet"]].copy()
    out["cluster"] = labels
    return out


def cluster_evidence(features: pd.DataFrame, ranked: pd.DataFrame | None = None,
                     eps: float = DEFAULT_EPS,
                     min_samples: int = DEFAULT_MIN_SAMPLES) -> pd.Series:
    """Label-free per-wallet cluster signal for risk_model's 3c 'cluster
    evidence': 1.0 if DBSCAN put the wallet in a distinct behavioural group
    (a non-noise cluster that is not the single largest / mainstream cluster),
    else 0.0 -- then z-scored.

    Deliberately NOT derived from the anomaly score (that would just be a noisy
    copy of ai_evidence and the logistic regression coefficients would become
    uninterpretable). `ranked` is accepted for signature symmetry but unused.
    """
    _ = ranked
    clusters = cluster_wallets(features, eps, min_samples).set_index("wallet")["cluster"]
    non_noise = clusters[clusters >= 0]
    mainstream = non_noise.value_counts().idxmax() if len(non_noise) else None

    raw = clusters.apply(
        lambda c: 1.0 if (c >= 0 and c != mainstream) else 0.0
    )
    std = raw.std(ddof=0)
    return (raw - raw.mean()) / std if std else pd.Series(0.0, index=raw.index)


def purity_report(clusters: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Per cluster: size, anomalous/normal split, and the dominant anomaly type."""
    m = clusters.merge(labels, left_on="wallet", right_on="entity", how="left")
    rows = []
    for cid, grp in m.groupby("cluster"):
        anom = grp[grp["label"] == "anomalous"]
        types = anom["anomaly_type"].value_counts()
        rows.append(
            {
                "cluster": cid,
                "size": len(grp),
                "n_anomalous": len(anom),
                "n_normal": int((grp["label"] == "normal").sum()),
                "frac_anomalous": round(len(anom) / len(grp), 3),
                "dominant_type": (types.index[0] if len(types) else "-"),
                "dominant_type_n": (int(types.iloc[0]) if len(types) else 0),
            }
        )
    df = pd.DataFrame(rows).sort_values(
        ["frac_anomalous", "size"], ascending=[False, False]
    ).reset_index(drop=True)
    return df


def run(tx_csv: str, labels_csv: str, eps: float = DEFAULT_EPS,
        min_samples: int = DEFAULT_MIN_SAMPLES):
    labels = pd.read_csv(labels_csv)
    df = pipeline.load_transactions(tx_csv)
    g = pipeline.build_graph(df)
    features = pipeline.compute_wallet_features(df, g)
    clusters = cluster_wallets(features, eps, min_samples)
    report = purity_report(clusters, labels)
    return clusters, report, features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tx", default="data/synthetic_transactions.csv")
    ap.add_argument("--labels", default="data/synthetic_labels.csv")
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS)
    ap.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    args = ap.parse_args()

    clusters, report, features = run(args.tx, args.labels, args.eps, args.min_samples)

    X = _scaled_matrix(features)
    print(f"DBSCAN eps={args.eps}  min_samples={args.min_samples}  "
          f"(k-distance percentiles: {_kdistance_percentiles(X, args.min_samples)})")
    n_clusters = clusters[clusters["cluster"] >= 0]["cluster"].nunique()
    n_noise = int((clusters["cluster"] == -1).sum())
    print(f"{n_clusters} clusters + {n_noise} noise points, {len(clusters)} wallets total\n")

    print(f"{'cluster':>8s} {'size':>6s} {'anom':>6s} {'norm':>6s} {'frac_anom':>10s}  dominant type")
    for _, r in report.iterrows():
        tag = "NOISE" if r["cluster"] == -1 else f"#{r['cluster']}"
        dom = f"{r['dominant_type']} ({r['dominant_type_n']})" if r["dominant_type_n"] else "-"
        print(f"{tag:>8s} {r['size']:6d} {r['n_anomalous']:6d} {r['n_normal']:6d} "
              f"{r['frac_anomalous']:10.3f}  {dom}")

    pure = report[(report["cluster"] >= 0) & (report["frac_anomalous"] >= 0.8)]
    print(f"\n{len(pure)} clusters are >=80% anomalous "
          f"({int(pure['n_anomalous'].sum())} anomalous wallets grouped); "
          "these are DBSCAN actually grouping related actors, not noise.")


if __name__ == "__main__":
    main()
