# BitGuard AI — demo script (SIH26146)

~5 minutes. Walks **one** flagged entity from a raw metadata row through to a
finished investigative lead. Every command runs offline.

Subject: wallet `1WYEQq1zGSyMaF46a5hN8yQnDx0340` — a wallet on a planted
**circular-flow** ring in the synthetic dataset (we control the ground truth,
so we know the right answer before the system produces it).

---

## 0. One-time setup (before the demo, with internet)

```bash
pip install -r requirements.txt
python scripts/setup_geoip.py          # fetch the DB-IP Lite GeoIP database
python src/generate_dataset.py --out-dir data --seed 7
```

The dataset ships in the repo; regenerate only to show it is deterministic.
Prints: `243 transactions`, `349 labelled entities` (280 normal, 69 anomalous,
3 instances each of six laundering patterns). Nothing after this needs network.

---

## 1. The raw input — "what the analyst is handed"

Bulk transaction + network metadata, the schema from the problem statement.
Show the six rows of this ring (from `data/synthetic_transactions.csv`):

```
timestamp             src_ip           txid     input -> output          geo  asn
2026-09-03 17:23:00   88.198.82.91     tx00137  1xNpddV… -> 1PQZ8C…      DE   AS24940  (Hetzner)
2026-09-03 17:30:00   13.228.143.79    tx00138  1PQZ8C…  -> 1wpWmo…      SG   AS16509  (Amazon)
2026-09-03 17:38:00   15.188.33.110    tx00139  1wpWmo…  -> 1rqYWc…      FR   AS16509  (Amazon)
2026-09-03 17:52:00   8.8.8.141        tx00140  1rqYWc…  -> 1XNjSp…      US   AS15169  (Google)
2026-09-03 18:01:00   13.228.46.34     tx00141  1XNjSp…  -> 1WYEQq…      SG   AS16509  (Amazon)
2026-09-03 18:05:00   178.154.236.244  tx00142  1WYEQq…  -> 1xNpddV…     RU   AS200350 (Yandex)
```

The `geo` and `asn` columns are **not hand-typed** — each `src_ip` is a real
public address and those values come from an actual lookup against the bundled
GeoIP database (`python src/geoip.py` shows the lookups directly). By eye: six
hops, five countries, ~40 minutes, and the funds return to the wallet they
started from (`1xNpddV…`). No single row shows that.

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
- **Learned weights** (spec 3c): `graph 0.36, cluster 0.30, ai 0.24, behavioral 0.10`
  — fitted from the labelled data by logistic regression, **not** hand-picked
  40/20/20/20. The comparison table shows circular-flow wallets moving up the
  ranking versus the naive split.
- The top of the risk table is circular-flow wallets. Each shows **Risk** and
  **Confidence** as separate columns, and they are allowed to disagree.

## 4. Clustering — corroboration, not a second verdict

```bash
python src/clustering.py
```

Point at: 4 clusters come out **≥80% anomalous** (57 of 69 anomalous wallets
grouped by pattern); the 235-wallet mainstream blob is **0%**. DBSCAN is
grouping related actors. Supporting evidence for the model, never an independent
anomaly call (spec section 4).

## 5. The investigative lead

```bash
python src/case_file.py --wallets 1WYEQq1zGSyMaF46a5hN8yQnDx0340
```

Read the whole record aloud — this is the deliverable (spec 3e):

```
INVESTIGATIVE LEAD -- 1WYEQq1zGSyMaF46a5hN8yQnDx0340
Risk score        : 98.2 / 100   (Critical)
Confidence score  : 0.90        (corr 0.82 / completeness 1.00)
  Risk = how unusual the behaviour is.  Confidence = how much to trust the evidence.

WHY FLAGGED (behaviour warranting investigation -- not a conclusion):
  - [rapid_movement] forwarded funds 4.0 min after receiving them
  - [layering] sits on a 11-wallet pass-through chain
  - [circular_flow] funds leaving this wallet return to it after 6 hops

SUPPORTING EVIDENCE -- transactions: tx00141, tx00142
TIMELINE:
  2026-09-03 18:01  received via tx00141  -> 1XNjSpsxjfVF
  2026-09-03 18:05  sent     via tx00142  -> 1xNpddVhHmSk
INVESTIGATION PATH:
  IP 178.154.133.195 -> TXID tx00142 -> Wallet 1WYEQq1z… -> Wallet 1xNpddV… -> Wallet 1PQZ8C… -> Wallet 1wpWmo…
```

Note the language: *"behaviour warranting investigation,"* never *"identifies a
criminal"* or *"proves laundering."*

## 6. The dashboard — "what the judge sees"

```bash
streamlit run src/dashboard.py
```

1. **Command Center** — Transactions 243 · Entities 349 · Model-flagged 88 ·
   Behaviour clusters 8 · High/Critical 67.
2. **Priority Alerts** — ranked table; circular-flow wallets at the top.
3. Pick `1WYEQq1zGSyMaF46a5hN8yQnDx0340` in the **Investigative Lead** dropdown
   → the full case file from step 5 renders next to the entity graph
   (red = flagged) and the DBSCAN purity table.

## 7. Prove it is offline

```bash
python scripts/offline_selfcheck.py
```

Installs a socket guard that raises on any outbound connection, then runs all
10 pipeline stages — including the GeoIP lookups. Ends with `ALL 10 STAGES RAN
OFFLINE`. For a physical check: disconnect the network and re-run any command
above (setup_geoip aside) — they all still work.

---

## The one-line story

> A wallet moved money through five other wallets across five countries in
> forty minutes and got it all back. No single record shows that. Correlating
> the network and blockchain layers into a graph — with the geo derived from a
> real GeoIP lookup — running a trained anomaly model, and corroborating with
> clustering surfaces it as a ranked, explainable, separately-confidence-scored
> lead, fully offline.

## Honest limitations (say these before a judge asks)

- Synthetic data only; the real NTRO dataset has not been swapped in.
- `layering` recall by the raw model flag is ~0.3 — the feature is
  discriminative and the risk score catches these wallets, but the Isolation
  Forest under-flags long-chain interior wallets. Known, documented.
- The Isolation Forest flags ~15% of normal wallets; on this dataset ~12 reach
  the High band. The learned risk score's scaling is dataset-sensitive (across
  seeds this swings ~0–14), so the flag rate and the band thresholds still need
  tuning against held-out data.
- All thresholds (risk buckets, DBSCAN eps, correlation accept threshold) are
  prototype values, not tuned against anything official.
