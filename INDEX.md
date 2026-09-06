# INDEX.md — start here if you're reading this cold

This repo is an offline prototype for SIH26146 (NTRO): it ingests bulk Bitcoin
transaction + network metadata, correlates the network and blockchain layers
into one graph, runs a single Isolation Forest over 9 wallet features to flag
anomalies, corroborates with DBSCAN clustering, scores each entity with a
**learned** risk weight and a **separately computed** confidence, and emits a
structured investigative-lead case file — all viewable in a minimal Streamlit
dashboard and provably offline. It was built over four working sessions
("nights"); each night's `NIGHTn_NOTES.md` is a self-contained plain-language
record of what was built, what was tested with real output, and what was left
open. Start with `README.md` for the current state, then `DEMO.md` for a
narrated single-case walk-through.

## Reading order

1. **`README.md`** — current state, the pipeline table, honest `[x]/[ ]` status.
2. **`DEMO.md`** — one planted circular-flow wallet from four raw rows to a
   finished lead; every command shown.
3. **`SIH_BUILD_SPEC.md`** — the original scope and constraints (sections 3a–3e,
   the build order, "never overclaim", "one primary ML model").
4. **`NIGHT1_NOTES.md`** — the labelled synthetic **generator** (spec 3b) and
   **correlation confidence** scoring (spec 3a). Also: the original fan-out bug.
5. **`NIGHT2_NOTES.md`** — the three **multi-hop features**, wiring correlation
   confidence into `build_graph`, **learned risk weights** (3c) and the
   **risk/confidence split** (3d).
6. **`NIGHT3_NOTES.md`** — **the cycle-hop false-positive bug and its fix**
   (`min_return_cycle_hops` capped 12→7; it was a model FP, not a labelling
   bug); **DBSCAN clustering** + purity; the **six pattern heuristics**; the
   **investigative-lead case file** format (3e).
7. **`NIGHT4_NOTES.md`** — the **Streamlit dashboard**, the **offline
   self-check**, the demo script. Ends with the final open-items list and the
   Sept 10 polish checklist.
8. **`PPT_OUTLINE.md`** — 15 slides, each wired to the script/command/output
   that is its evidence.

## Source map (`src/`)

| file | what | notes file |
|---|---|---|
| `main.py` | ingestion, `build_graph` (+ correlation-scored edges), `compute_wallet_features` (9 features), `flag_anomalies` (Isolation Forest), `_explain_row`, static graph PNG | 1, 2, 3 |
| `generate_dataset.py` | labelled synthetic generator, six planted patterns, deterministic; `src_ip` from real prefixes, geo/asn by real GeoIP lookup | 1 |
| `geoip.py` | `GeoIPResolver` — IP → country / ASN against the local DB-IP Lite `.mmdb` files, offline | — |
| `correlation_confidence.py` | `score_correlation` — `0.5·time + 0.3·port + 0.2·ambiguity`, `ACCEPTED`/`UNRESOLVED` | 1, 2 |
| `diagnostics.py` | per-anomaly-type recall of the current pipeline | 2, 3 |
| `clustering.py` | DBSCAN + purity report + label-free `cluster_evidence` | 3 |
| `risk_model.py` | 4 evidence signals → logistic regression → risk weights (3c); disjoint confidence (3d); `score_entities` | 2, 3 |
| `pattern_heuristics.py` | the six laundering signals as named `PatternSignal`s | 3 |
| `case_file.py` | the investigative-lead record (3e) | 3 |
| `dashboard.py` | minimal Streamlit command center | 4 |

## Scripts & data

- `scripts/run_all.sh` — one command, runs the whole pipeline then the dashboard.
- `scripts/setup_geoip.py` — one-time fetch of the DB-IP Lite GeoIP `.mmdb` files.
- `scripts/offline_selfcheck.py` — socket-guarded full run; prints `ALL 10 STAGES RAN OFFLINE`.
- `data/sample_transactions.csv` — 20-row smoke test (the README graph is from this).
- `data/synthetic_transactions.csv` / `synthetic_labels.csv` — the 243-tx labelled set (seed 7), regenerable; geo derived by real GeoIP lookup.
- `data/geoip/*.mmdb` — the bundled GeoIP databases (DB-IP Lite, CC BY 4.0).
- `tests/` — 73 tests: `python -m pytest`.

## Known-open, in one place

Real NTRO dataset not swapped in · `layering` raw-flag recall ~0.3 (feature +
clustering + risk score do catch these; the IF flag alone doesn't) · the
learned risk score's scaling is dataset-sensitive, so the number of normal
wallets in the High band swings ~0–14 across seeds (~12 on the committed set) ·
no threshold tuning against held-out data. Details: `README.md` status list and
each night's "What is NOT done". (GeoIP enrichment is now done — `src/geoip.py`.)
