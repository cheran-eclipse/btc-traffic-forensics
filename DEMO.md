# BitGuard AI — demo script (SIH26146)

~5 minutes. Walks **one** flagged entity from a raw metadata row through to a
finished investigative lead. Every command runs offline.

Subject: wallet `1fAJSeQYmUnwYfftUrJsW6dLxX0341` — a planted **circular-flow**
case in the synthetic dataset (we control the ground truth, so we know the
right answer before the system produces it).

---

## 0. One-time setup (before the demo, with internet)

```bash
pip install -r requirements.txt
python src/generate_dataset.py --out-dir data --seed 7
```

Prints: `229 transactions`, `354 labelled entities` (293 normal, 61 anomalous,
3 each of six laundering patterns). The rest of the demo needs no network.

---

## 1. The raw input — "what the analyst is handed"

Bulk transaction + network metadata, the schema from the problem statement.
Show these four rows (from `data/synthetic_transactions.csv`):

```
timestamp             src_ip           txid     input -> output                              geo
2026-09-01 17:15:00   226.120.18.223   tx00028  1fAJSe… -> 1pS4ae…                            DE
2026-09-01 17:30:00   49.48.131.136    tx00030  1pS4ae… -> 1PMFbQ…                            IN
2026-09-01 17:44:00   171.158.3.91     tx00032  1PMFbQ… -> 11CPLo…                            GB
2026-09-01 17:51:00   61.102.251.86    tx00033  11CPLo… -> 1fAJSe…                            US
```

By eye: four hops, four countries, 36 minutes, and the funds end up **back at
the wallet they started from**. Nothing in a single row says that — it is only
visible once the two layers are correlated into a graph.

---

## 2. Run the pipeline

```bash
python src/main.py --input data/synthetic_transactions.csv --top 10
```

Point at: `[2/5] Building IP <-> wallet <-> tx correlation graph` — network and
blockchain layers are now one graph. The Isolation Forest (the required AI/ML
core) scores every wallet; explanations are attached in plain language.

## 3. Risk vs. Confidence — two numbers, never merged

```bash
python src/risk_model.py
```

Point at:
- **Learned weights** (spec 3c): `graph 0.42, ai 0.24, cluster 0.23, behavioral 0.11`
  — fitted from the labelled data by logistic regression, **not** hand-picked
  40/20/20/20. In the comparison table the two normal-wallet false positives
  drop **14 rank places** under the fitted weights versus the naive split
  (`1PSEeV8p… 58.7 fitted vs 87.0 naive`).
- The top of the risk table is circular-flow wallets (RISK 77–100). Each shows
  **Risk** and **Confidence** as separate columns — e.g. a peeling-chain wallet
  at RISK 79.3 but CONF 0.83.

## 4. Clustering — corroboration, not a second verdict

```bash
python src/clustering.py
```

Point at: 5 clusters come out **100% anomalous**; the 248-wallet mainstream
blob is **0%**. DBSCAN is grouping related actors. This is supporting evidence
for the model, never an independent anomaly call (spec section 4).

## 5. The investigative lead

```bash
python src/case_file.py --wallets 1fAJSeQYmUnwYfftUrJsW6dLxX0341
```

Read the whole record aloud — this is the deliverable (spec 3e):

```
INVESTIGATIVE LEAD -- 1fAJSeQYmUnwYfftUrJsW6dLxX0341
Risk score        : 53.3 / 100   (Medium)
Confidence score  : 0.91        (corr 0.86 / completeness 1.00)
  Risk = how unusual the behaviour is.  Confidence = how much to trust the evidence.

WHY FLAGGED (behaviour warranting investigation -- not a conclusion):
  - [layering] sits on a 7-wallet pass-through chain
  - [circular_flow] funds leaving this wallet return to it after 4 hops

SUPPORTING EVIDENCE -- transactions: tx00028, tx00033
TIMELINE:
  2026-09-01 17:15  sent     via tx00028  -> 1pS4aed62dUr
  2026-09-01 17:51  received via tx00033  -> 11CPLoh4RhyS
INVESTIGATION PATH:
  IP 226.120.18.223 -> TXID tx00028 -> Wallet 1fAJSe… -> Wallet 1pS4ae… -> Wallet 1PMFbQ… -> Wallet 11CPLo…
```

Note the language: *"behaviour warranting investigation,"* never *"identifies a
criminal"* or *"proves laundering."*

## 6. The dashboard — "what the judge sees"

```bash
streamlit run src/dashboard.py
```

1. **Command Center** — Transactions 229 · Entities 354 · Model-flagged 81 ·
   Behaviour clusters 8 · High/Critical 20.
2. **Priority Alerts** — ranked table; circular-flow wallets at the top.
3. Pick `1fAJSeQYmUnwYfftUrJsW6dLxX0341` in the **Investigative Lead** dropdown
   → the full case file from step 5 renders next to the entity graph
   (red = flagged) and the DBSCAN purity table.

## 7. Prove it is offline

```bash
python scripts/offline_selfcheck.py
```

Installs a socket guard that raises on any outbound connection, then runs all
10 pipeline stages. Ends with `ALL 10 STAGES RAN OFFLINE`. For a physical
check: disconnect the network and re-run any command above — they all still
work.

---

## The one-line story

> A wallet moved money through four other wallets in four countries in 36
> minutes and got it all back. No single record shows that. Correlating the
> network and blockchain layers into a graph, running a trained anomaly model,
> and corroborating with clustering surfaces it as a ranked, explainable,
> separately-confidence-scored lead — fully offline.

## Honest limitations (say these before a judge asks)

- Synthetic data only; the real NTRO dataset has not been swapped in.
- `layering` recall by the raw model flag is ~0.35 — the feature is
  discriminative and the risk score catches these wallets, but the Isolation
  Forest under-flags long-chain interior wallets. Known, documented.
- ~2 normal wallets still score high on accidental long chains in the
  random-payment mesh.
- All thresholds (risk buckets, DBSCAN eps, correlation accept threshold) are
  prototype values, not tuned against anything official.
