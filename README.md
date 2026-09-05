# btc-traffic-forensics

**AI-powered correlation of network-layer and blockchain-layer data to surface
investigative leads from Bitcoin transaction traffic.**

Built for **SIH 2026 · SIH26146**, sponsored by the **National Technical
Research Organisation (NTRO)** (Theme: Blockchain & Cybersecurity).

---

## Author

**Cheran N K** — B.Tech Electronics & Communication Engineering (Final Year),
Puducherry Technological University
[github.com/cheran-eclipse](https://github.com/cheran-eclipse) · cheran.space@gmail.com

## Why this exists

Bitcoin's pseudonymous, peer-to-peer design lets illicit funds move, layer, and
cash out — ransomware payoffs, darknet proceeds, extortion — while evading
traditional financial surveillance. The challenge is to build an **offline**
system that ingests bulk transaction + network metadata, correlates the two
layers, applies a real trained model (not hand-written rules) to rank which
wallets are worth an investigator's time, and attaches an explanation and a
confidence — not just a score.

## One-command run

```bash
pip install -r requirements.txt
bash scripts/run_all.sh
```

Generates the labelled dataset, runs the full pipeline, prints the per-pattern
diagnostic, the learned risk weights, the cluster purity report and one full
investigative-lead case file, then launches the dashboard. Everything after the
`pip install` runs with no network. See [`DEMO.md`](DEMO.md) for a narrated
5-minute walk-through of a single flagged case, and [`INDEX.md`](INDEX.md) for a
reading order through the code and the design notes.

## Pipeline

| stage | module | what it does |
|---|---|---|
| **Labelled data** | `src/generate_dataset.py` | Synthesises transactions **and** a ground-truth label per wallet by planting six laundering patterns (peeling chain, high velocity, rapid movement, amount splitting, layering, circular flow). Deterministic per seed. A generator, not a detector — it's what lets the risk weights be *learned* instead of guessed. Committed output: `data/synthetic_transactions.csv` (229 tx) + `data/synthetic_labels.csv` (354 entities, 61 anomalous). |
| **Correlation graph** | `src/main.py :: build_graph` | One heterogeneous graph of IPs, wallets and transactions. Every `ip → tx` edge carries a **correlation confidence** (`0.5·time + 0.3·port + 0.2·ambiguity`, from `src/correlation_confidence.py`) and an `ACCEPTED` / `UNRESOLVED` status. A match in a crowded time window scores lower; a weak one is left `UNRESOLVED`, never guessed. An accepted edge is an *observation*, never proof of wallet ownership. |
| **Features** | `src/main.py :: compute_wallet_features` | 9 features per wallet. Six are local (fan-in, fan-out, distinct countries, tx count, activity span, max tx output fan-out). Three are **multi-hop**, added because the local six were blind to whole pattern classes: <br>• `min_return_cycle_hops` — fewest hops for funds to return to a wallet (**circular flow**; a ring is a path several hops away, invisible to any single wallet's degree). Search is capped at 7 hops — deeper matches are just the random-payment mesh, not laundering. <br>• `linear_chain_length` — length of the 1-in/1-out pass-through chain through a wallet (**layering**; each hop wallet looks perfectly normal locally). <br>• `min_receive_to_forward_minutes` — shortest hold time before forwarding funds on (**rapid movement / mule**; total activity span actively hides this). |
| **AI/ML detection** | `src/main.py :: flag_anomalies` | One Isolation Forest over the 9 scaled features — the single ML core the PS requires. Produces an anomaly score and a plain-language reason per wallet. |
| **Clustering** | `src/clustering.py` | DBSCAN over the same features. Reports per-cluster purity: 5 clusters come out 100 % anomalous (49 of 61 anomalous wallets grouped by pattern), the 248-wallet mainstream blob 0 %. Supporting evidence, never a competing verdict. |
| **Risk (learned)** | `src/risk_model.py` | Four evidence signals per entity — AI anomaly score, graph evidence, behavioural evidence, cluster evidence — fed to a logistic regression fit against the labels. The fitted coefficients **are** the risk weights: `graph 0.42, ai 0.24, cluster 0.23, behavioral 0.11` — not a hand-picked 40/20/20/20. Risk is bucketed Low / Medium / High / Critical (prototype thresholds). |
| **Confidence (separate)** | `src/risk_model.py` | A second number from *disjoint* evidence: mean correlation confidence of the entity's network edges + feature completeness. **Risk and Confidence are never merged** — a lead can be high-risk / low-confidence, and that disagreement is information. |
| **Pattern heuristics** | `src/pattern_heuristics.py` | The six laundering patterns as named, explainable `PatternSignal`s (fired / not, why, which TXIDs) — supporting evidence for the case file, never a fifth verdict. |
| **Investigative lead** | `src/case_file.py` | Per flagged entity: risk + bucket, confidence + breakdown, why-flagged reasons, supporting TXIDs, related wallets/IPs, time-ordered timeline, `IP → TXID → Wallet → Wallet` investigation path. |
| **Dashboard** | `src/dashboard.py` | Minimal Streamlit: Command Center counts, ranked Priority Alerts, and the full case file per selected lead beside the entity graph. `streamlit run src/dashboard.py`. |
| **Offline proof** | `scripts/offline_selfcheck.py` | Runs all 10 stages behind a socket guard that raises on any outbound connection. Ends `ALL 10 STAGES RAN OFFLINE`. |

![entity graph](assets/entity_graph_demo.png)
*From `data/sample_transactions.csv`. Red = flagged wallets, blue = wallets,
green = IPs, grey = transactions.*

## Status — what's real vs. what's open

- [x] Schema ingestion, IP↔wallet↔tx correlation graph, offline end to end
- [x] Correlation confidence per network edge (`ACCEPTED` / `UNRESOLVED`), never claims ownership
- [x] 9-feature detector with one Isolation Forest; plain-language reasons
- [x] Labelled synthetic dataset generator with known ground truth
- [x] DBSCAN clustering with a purity report (5 clusters 100 % anomalous)
- [x] Learned risk weights (logistic regression), not hand-picked
- [x] Risk and Confidence as two separately-computed, separately-reported numbers
- [x] Six named pattern heuristics as supporting evidence
- [x] Investigative-lead case file format
- [x] Minimal command-center dashboard
- [x] Offline self-check; `requirements-lock.txt` for a reproducible install
- [x] 60 tests (`python -m pytest`)
- [ ] **Real NTRO dataset** swapped in for the synthetic one
- [ ] **GeoIP enrichment** — currently reads geo/ASN columns from the data rather than deriving them
- [ ] **`layering` recall by the raw model flag is ~0.35** — the feature separates layering wallets cleanly and clustering + the risk score catch them, but the Isolation Forest itself under-flags long-chain interior wallets
- [ ] **False positives** — the Isolation Forest flags ~14 % of normal wallets; none reach High/Critical risk (highest normal risk ~59, i.e. Medium), so they surface as Medium-severity noise rather than false leads, but the raw model flag rate is still higher than it should be
- [ ] **Threshold tuning** — risk buckets, DBSCAN `eps`, correlation accept threshold and the six heuristic thresholds are all prototype values, not tuned against held-out data
- [ ] Packaging for a clean machine beyond the lock file

### The fan-out bug, kept on purpose

The first version of the fan-out feature measured a wallet's own out-degree in
the graph — the wrong thing. A peeling chain is a property of the *transaction*
(one input splitting into many new outputs), not an edge count on the input
wallet. Caught by checking the planted peeling-chain wallet and seeing it wasn't
flagged for the right reason; fixed as `max_tx_output_fanout`, re-ran,
confirmed. Left documented rather than quietly patched — the same discipline
(verify a new feature against a known case before trusting it) caught the Night 3
cycle-hop bug too.

## Setup

```bash
pip install -r requirements.txt            # or requirements-lock.txt for exact versions
python src/main.py --input data/sample_transactions.csv --top 10
```

Tests: `pip install -r requirements-dev.txt && python -m pytest`

## License

MIT — see [LICENSE](LICENSE).
