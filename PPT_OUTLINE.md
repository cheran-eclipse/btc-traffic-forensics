# PPT_OUTLINE.md — 15 slides, each wired to evidence that already exists

This maps the standard SIH prototype-pitch structure (15 slides) to the
artefacts already in this repo. **No new content** — every slide below names
the exact script, command, or output file that produces its evidence, so
building the deck is assembly, not creation. If the BitGuard AI master doc
(Part 4, §28) numbers slides differently, renumber — the evidence wiring is
the point.

Regenerate everything first with: `bash scripts/run_all.sh --no-dashboard`

---

### 1 — Title
Team, PS **SIH26146** (NTRO, Blockchain & Cybersecurity), project name.
Evidence: `README.md` header; author block.

### 2 — Problem
Bitcoin lets illicit funds layer and cash out while evading financial
surveillance; investigators need ranked, explainable leads from bulk metadata,
offline.
Evidence: `README.md` "Why this exists"; `SIH_BUILD_SPEC.md` §1.

### 3 — Solution overview
One offline pipeline: correlate network + blockchain layers → trained anomaly
model → clustering → learned risk + separate confidence → investigative-lead
case file → dashboard.
Evidence: `README.md` "Pipeline" table (screenshot it); architecture diagram
to be drawn from that table.

### 4 — Innovation / what's different
- Risk weights **learned by logistic regression from labelled synthetic
  ground truth**, not hand-picked 40/20/20/20.
- **Risk and Confidence are two separate numbers** from disjoint evidence.
- Correlation edges carry a real confidence; weak matches are `UNRESOLVED`,
  never guessed.
- Never overclaims — "behaviour warranting investigation", never "proves".
Evidence: `src/risk_model.py` output (weights table); `src/correlation_confidence.py`;
`NIGHT2_NOTES.md` (3c/3d), `NIGHT1_NOTES.md` (3a).

### 5 — Architecture
The 11-stage table in `README.md`, as a left-to-right diagram:
`generate_dataset → build_graph (+correlation_confidence) → compute_wallet_features
→ flag_anomalies (Isolation Forest) → clustering (DBSCAN) → risk_model (LogReg)
→ pattern_heuristics → case_file → dashboard`, with `offline_selfcheck` wrapping all.
Evidence: `README.md` Pipeline table; `INDEX.md` source-file map.

### 6 — Tech stack
Python 3, pandas / numpy, scikit-learn (IsolationForest, DBSCAN,
LogisticRegression), networkx, matplotlib, Streamlit. Fully offline at runtime.
Evidence: `requirements.txt`, `requirements-lock.txt` (54 pinned packages).

### 7 — Data: labelled synthetic generator (spec 3b)
Plants six laundering patterns and records a ground-truth label per wallet.
229 transactions, 354 entities (293 normal / 61 anomalous), deterministic.
Command: `python src/generate_dataset.py --out-dir data --seed 7`
Evidence: `src/generate_dataset.py`; `data/synthetic_transactions.csv`,
`data/synthetic_labels.csv`; `NIGHT1_NOTES.md` §1.

### 8 — Correlation: two layers, one graph (spec 3a)
IP ↔ wallet ↔ tx graph; every `ip→tx` edge scored
`0.5·time + 0.3·port + 0.2·ambiguity` → `ACCEPTED` / `UNRESOLVED`.
Command: `python src/correlation_confidence.py` (reference cases)
Evidence: `src/correlation_confidence.py`, `src/main.py :: build_graph`;
`NIGHT1_NOTES.md` §2, `NIGHT2_NOTES.md` step 2.

### 9 — Detection: features + one ML model
9 features (6 local + 3 multi-hop: `min_return_cycle_hops`,
`linear_chain_length`, `min_receive_to_forward_minutes`) → one Isolation Forest
→ anomaly score + plain-language reason.
Command: `python src/main.py --input data/synthetic_transactions.csv --top 10`
Diagnostic: `python src/diagnostics.py` (per-pattern recall — show the table,
including layering 0.35 honestly).
Evidence: `src/main.py`, `src/diagnostics.py`; `NIGHT2_NOTES.md` step 1;
`NIGHT3_NOTES.md` (cycle-hop fix).

### 10 — Clustering: corroboration (module 7)
DBSCAN over the same features. 5 clusters 100 % anomalous (49/61 grouped);
248-wallet mainstream blob 0 %. Supporting evidence, not a competing verdict.
Command: `python src/clustering.py`
Evidence: `src/clustering.py`; `NIGHT3_NOTES.md` §1 (purity table).

### 11 — Risk (learned) + Confidence (separate) — spec 3c / 3d
Four evidence signals → logistic regression → weights
`graph 0.42, ai 0.24, cluster 0.23, behavioral 0.11` vs naive `0.40/0.20/0.20/0.20`.
Risk bucketed Low/Med/High/Critical (prototype). Confidence from disjoint
evidence; the two may disagree.
Command: `python src/risk_model.py` (weights table + fitted-vs-naive comparison
+ Risk/Confidence table)
Evidence: `src/risk_model.py`; `NIGHT2_NOTES.md` step 3/4, `NIGHT3_NOTES.md` §1.

### 12 — Output: the investigative-lead case file (spec 3e)
Entity, Risk + bucket, Confidence + breakdown, why-flagged, supporting TXIDs,
related entities, timeline, `IP → TXID → Wallet → Wallet` path.
Command: `python src/case_file.py --wallets 1fAJSeQYmUnwYfftUrJsW6dLxX0341`
Evidence: `src/case_file.py`; full worked example in `NIGHT3_NOTES.md` §3 and
`DEMO.md` step 5 — paste that block onto the slide.

### 13 — Dashboard / live demo
Command Center counts · ranked Priority Alerts · case file + entity graph per
selected lead.
Command: `streamlit run src/dashboard.py`
Evidence: `src/dashboard.py`; screenshot for the slide; `DEMO.md` step 6 is the
click sequence.

### 14 — Feasibility, offline constraint, honest limitations
- Offline: `python scripts/offline_selfcheck.py` → `ALL 10 STAGES RAN OFFLINE`.
- Reproducible: `requirements-lock.txt`, `bash scripts/run_all.sh`.
- Open: real NTRO dataset not yet swapped in; GeoIP not derived; `layering`
  raw-flag recall ~0.35; ~2 residual normal FPs; thresholds untuned.
Evidence: `scripts/offline_selfcheck.py`; `README.md` status list;
`NIGHT4_NOTES.md` "What is NOT done".

### 15 — Impact & next steps
Impact: investigator time triaged by a ranked, explainable, confidence-scored
lead list, offline on seized data. Next: real dataset, GeoIP enrichment,
threshold tuning on held-out data, expand the dashboard's link analysis.
Evidence: `README.md` status list; `NIGHT4_NOTES.md` Sept 10 checklist +
"Not built at all" list.

---

## One-liner per slide for the notes column

| # | slide | run this |
|---|---|---|
| 7 | data | `python src/generate_dataset.py --out-dir data --seed 7` |
| 8 | correlation | `python src/correlation_confidence.py` |
| 9 | detection | `python src/main.py --input data/synthetic_transactions.csv --top 10` ; `python src/diagnostics.py` |
| 10 | clustering | `python src/clustering.py` |
| 11 | risk | `python src/risk_model.py` |
| 12 | case file | `python src/case_file.py --wallets 1fAJSeQYmUnwYfftUrJsW6dLxX0341` |
| 13 | dashboard | `streamlit run src/dashboard.py` |
| 14 | offline | `python scripts/offline_selfcheck.py` |
| all | everything | `bash scripts/run_all.sh` |
