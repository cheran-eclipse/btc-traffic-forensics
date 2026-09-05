"""
SIH26146 / BitGuard AI -- sections 3c and 3d.

3c: risk weights are *learned*, not hand-picked. We compute four evidence
signals per entity, fit a logistic regression from those signals to the known
Night-1 labels, and use the fitted coefficients as the risk-score weights. The
spec is explicit that hand-picking something like 40/20/20/20 defeats the
purpose -- we own the synthetic ground truth, so the weights should come from
fitting it.

    ai_evidence         Isolation Forest anomaly score (the primary ML model)
    graph_evidence      structural signals: fan-in/out, tx output fan-out,
                        return-cycle hops, pass-through chain length
    behavioral_evidence velocity, geographic spread, hold-time-before-forwarding
    cluster_evidence    stubbed at 0.0 -- clustering is Night 3

3d: Risk and Confidence are two separate numbers, computed from disjoint
evidence, reported side by side and never collapsed into one.

    Risk       how unusual the behaviour is (the four signals above)
    Confidence how much we trust the evidence behind that judgement
               (mean correlation confidence of the entity's network edges +
                how complete its feature data is)

Risk buckets (0-30 Low, 31-60 Medium, 61-80 High, 81-100 Critical) are
PROTOTYPE thresholds, not official NTRO values.

Offline: numpy / pandas / scikit-learn only.

Run:
    python src/risk_model.py --tx data/synthetic_transactions.csv \
                             --labels data/synthetic_labels.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import main as pipeline

# 3c evidence-signal definitions. Sign is chosen so that larger == more
# suspicious for every constituent (some are negated below).
_GRAPH_FEATURES = [
    "fan_in", "fan_out", "max_tx_output_fanout",
    "min_return_cycle_hops", "linear_chain_length",
]
_BEHAVIOURAL_FEATURES_POS = ["num_tx_touched", "distinct_countries"]
_BEHAVIOURAL_FEATURES_NEG = ["min_receive_to_forward_minutes"]  # low hold time = suspicious

_NAIVE_WEIGHTS = {"ai": 0.40, "graph": 0.20, "behavioral": 0.20, "cluster": 0.20}
_EVIDENCE_ORDER = ["ai_evidence", "graph_evidence", "behavioral_evidence", "cluster_evidence"]

RISK_BUCKETS = [(30, "Low"), (60, "Medium"), (80, "High"), (100, "Critical")]


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def compute_evidence_signals(ranked: pd.DataFrame) -> pd.DataFrame:
    """One row per wallet: the four 3c evidence signals, each z-scored so
    larger == more suspicious. cluster_evidence is a deliberate 0.0 stub."""
    f = ranked.set_index("wallet")

    ai = _zscore(f["anomaly_score"])

    graph_parts = [_zscore(f[c]) for c in _GRAPH_FEATURES]
    graph = pd.concat(graph_parts, axis=1).mean(axis=1)

    beh_parts = [_zscore(f[c]) for c in _BEHAVIOURAL_FEATURES_POS]
    beh_parts += [_zscore(-f[c]) for c in _BEHAVIOURAL_FEATURES_NEG]
    behavioral = pd.concat(beh_parts, axis=1).mean(axis=1)

    cluster = pd.Series(np.zeros(len(f)), index=f.index)  # STUB: clustering is Night 3

    return pd.DataFrame(
        {
            "ai_evidence": ai,
            "graph_evidence": graph,
            "behavioral_evidence": behavioral,
            "cluster_evidence": cluster,
        }
    ).reset_index().rename(columns={"index": "wallet"})


def fit_risk_weights(evidence: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Fit logistic regression: 4 evidence signals -> P(anomalous). Return the
    learned coefficients (these ARE the risk weights) plus the fitted model."""
    merged = evidence.merge(labels, left_on="wallet", right_on="entity", how="inner")
    X = merged[_EVIDENCE_ORDER].to_numpy()
    y = (merged["label"] == "anomalous").astype(int).to_numpy()

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)

    coef = dict(zip(["ai", "graph", "behavioral", "cluster"], model.coef_[0]))
    abs_sum = sum(abs(v) for v in coef.values()) or 1.0
    normalised = {k: abs(v) / abs_sum for k, v in coef.items()}
    return {
        "raw_coefficients": coef,
        "intercept": float(model.intercept_[0]),
        "normalised_weights": normalised,
        "model": model,
        "n_train": len(merged),
        "n_positive": int(y.sum()),
    }


def score_entities(
    tx_csv: str, labels_csv: str
) -> tuple[pd.DataFrame, dict]:
    """Full pass: run the pipeline, build evidence, fit weights, and return a
    per-entity table with SEPARATE Risk and Confidence columns (3d)."""
    labels = pd.read_csv(labels_csv)
    df = pipeline.load_transactions(tx_csv)
    g = pipeline.build_graph(df)
    features = pipeline.compute_wallet_features(df, g)
    ranked = pipeline.flag_anomalies(features)

    evidence = compute_evidence_signals(ranked)
    fit = fit_risk_weights(evidence, labels)

    # Both risk scores are a weighted sum of the same four min-max-scaled
    # signals, rescaled to 0-100. The ONLY difference is the weights:
    #   risk_fitted -> the logistic-regression coefficients (3c)
    #   risk_naive  -> a hand-picked 40/20/20/20
    # so any difference in the numbers is purely the effect of fitting.
    scaled = pd.DataFrame(
        {c: _minmax(evidence.set_index("wallet")[c]) for c in _EVIDENCE_ORDER}
    ).reindex(evidence["wallet"])
    S = scaled.to_numpy()

    coef = np.array([fit["raw_coefficients"][k] for k in ["ai", "graph", "behavioral", "cluster"]])
    risk_fitted = _minmax(pd.Series(S @ coef)).to_numpy() * 100.0

    naive = np.array([_NAIVE_WEIGHTS[k] for k in ["ai", "graph", "behavioral", "cluster"]])
    risk_naive = _minmax(pd.Series(S @ naive)).to_numpy() * 100.0

    # -- Confidence (3d): disjoint evidence -- correlation confidence + data completeness --
    conf = _entity_confidence(df, g, features)

    out = evidence.copy()
    out["risk_fitted"] = np.round(risk_fitted, 1)
    out["risk_naive"] = np.round(risk_naive, 1)
    out["risk_bucket"] = [_bucket(r) for r in out["risk_fitted"]]
    out = out.merge(conf, on="wallet", how="left")
    out = out.merge(labels[["entity", "label", "anomaly_type"]],
                    left_on="wallet", right_on="entity", how="left").drop(columns=["entity"])
    out = out.sort_values("risk_fitted", ascending=False).reset_index(drop=True)
    return out, fit


def _entity_confidence(df, g, features) -> pd.DataFrame:
    """Confidence = 0.6 * mean correlation confidence of the entity's network
    edges + 0.4 * feature completeness. PROTOTYPE weights. Disjoint from Risk."""
    feat_cols = pipeline.FEATURE_COLS
    completeness = (features[feat_cols].notna() & (features[feat_cols] != 0)).mean(axis=1)
    completeness = pd.Series(completeness.to_numpy(), index=features["wallet"])

    # mean confidence of network edges on the txs each wallet touches
    tx_conf: dict[str, list] = {}
    for u, v, e in g.edges(data=True):
        if e.get("kind") == "network_link":
            tx_conf.setdefault(v, []).append(e["confidence"])

    rows = []
    for w in features["wallet"]:
        touched = [n for n in list(g.successors(w)) + list(g.predecessors(w))
                   if g.nodes[n].get("kind") == "tx"]
        confs = [c for t in touched for c in tx_conf.get(t, [])]
        corr_conf = float(np.mean(confs)) if confs else 0.0
        c = 0.6 * corr_conf + 0.4 * float(completeness.get(w, 0.0))
        rows.append({"wallet": w, "confidence": round(c, 3),
                     "corr_confidence": round(corr_conf, 3),
                     "feature_completeness": round(float(completeness.get(w, 0.0)), 3)})
    return pd.DataFrame(rows)


def _bucket(risk: float) -> str:
    for hi, name in RISK_BUCKETS:
        if risk <= hi:
            return name
    return "Critical"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tx", default="data/synthetic_transactions.csv")
    ap.add_argument("--labels", default="data/synthetic_labels.csv")
    args = ap.parse_args()

    table, fit = score_entities(args.tx, args.labels)

    print("=== 3c: learned risk weights (logistic regression) ===")
    print(f"trained on {fit['n_train']} entities, {fit['n_positive']} anomalous")
    print(f"{'signal':14s} {'raw coef':>10s} {'|w| fitted':>12s} {'naive':>8s}")
    for k in ["ai", "graph", "behavioral", "cluster"]:
        print(f"{k:14s} {fit['raw_coefficients'][k]:10.3f} "
              f"{fit['normalised_weights'][k]:12.2f} {_NAIVE_WEIGHTS[k]:8.2f}")
    print(f"intercept      {fit['intercept']:10.3f}")

    print("\n=== 3c: fitted vs naive 40/20/20/20 on example wallets ===")
    anoms = table[table["label"] == "anomalous"].head(4)
    norms = table[table["label"] == "normal"]
    examples = pd.concat([anoms, norms.head(2), norms.tail(2)])
    print(f"{'wallet':16s} {'type':16s} {'risk_fit':>9s} {'risk_naive':>11s} {'rank_chg':>9s}")
    fit_rank = table["risk_fitted"].rank(ascending=False, method="min")
    naive_rank = table["risk_naive"].rank(ascending=False, method="min")
    for i, r in examples.iterrows():
        drank = int(naive_rank[i] - fit_rank[i])
        print(f"{r['wallet'][:16]:16s} {str(r['anomaly_type']):16s} "
              f"{r['risk_fitted']:9.1f} {r['risk_naive']:11.1f} {drank:+9d}")

    print("\n=== 3d: Risk and Confidence are separate numbers (top 8 by Risk) ===")
    print(f"{'wallet':16s} {'type':16s} {'RISK':>6s} {'bucket':>9s} {'CONF':>6s}  (corr / completeness)")
    for _, r in table.head(8).iterrows():
        print(f"{r['wallet'][:16]:16s} {str(r['anomaly_type']):16s} "
              f"{r['risk_fitted']:6.1f} {r['risk_bucket']:>9s} {r['confidence']:6.2f}"
              f"  ({r['corr_confidence']:.2f} / {r['feature_completeness']:.2f})")


if __name__ == "__main__":
    main()
