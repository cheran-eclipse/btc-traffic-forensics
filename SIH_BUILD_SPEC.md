# SIH26146 — BitGuard AI Build Spec

**Deadline: prototype needed by Sept 10. Today is Sept 6, ~3am. Read this whole file before touching any code.**

## 1. What this project is

Official problem statement SIH26146, National Technical Research Organisation (NTRO), theme Blockchain & Cybersecurity: build an **offline**, Linux-based system that ingests bulk Bitcoin transaction + network metadata (synthetic — no real seized data provided), correlates network-layer signals (IP/port/timing) with blockchain-layer signals (wallet/TXID/amount), applies genuine AI/ML anomaly detection (not just rules), clusters related entities, and outputs a ranked, explainable, confidence-scored list of investigative leads.

## 2. What already exists — READ THIS CODE FIRST, don't rewrite blind

This repo already has working, tested code:
- `src/main.py` — ingestion (CSV schema: timestamp, src_ip, dst_ip, src_port, dst_port, txid, input/output addresses+amounts, geo_country, asn), a NetworkX graph linking IPs/wallets/transactions, per-wallet feature extraction, an Isolation Forest anomaly model, plain-language explanation generation, and a static matplotlib graph visualization.
- `data/sample_transactions.csv` — 20-row synthetic sample with a planted 10-way peeling chain and a planted geo-hopping pattern, used to verify the detector actually catches known cases.
- `README.md` — documents a real bug that was caught and fixed: the first version of a "fan_out" feature measured the wrong thing (wallet's own out-degree, not the transaction's output count). Read this section before adding new graph features — same mistake is easy to repeat.

**Do not throw this away and start over.** Extend it. If you change existing function behavior, say so explicitly and explain why.

## 3. What's being added — refinements from a separate design document ("BitGuard AI")

These are real methodological upgrades, not just more features. Each one exists to fix a specific weakness:

### 3a. Correlation confidence scoring (upgrades the current IP↔tx graph edges)
Current code links every IP to every transaction it touches with no notion of confidence. Replace/augment with an explicit scoring function per network-observation-to-transaction candidate match:

```
confidence = 0.5 * time_score + 0.3 * port_score + 0.2 * ambiguity_penalty
```
- `time_score`: higher when observation and candidate transaction timestamps are closer together
- `port_score`: higher when the observed port matches the known Bitcoin P2P port in the dataset
- `ambiguity_penalty`: 1 / (number of candidate transactions in the same time window) — a match surrounded by many equally-plausible candidates should score lower

Matches above a threshold get accepted and labeled as an "observation" (never as proof of ownership). Matches below threshold get marked `UNRESOLVED`, not forced into a guess. **Why this matters**: a raw nearest-timestamp match silently picks one candidate when several cluster in the same window, and reports false certainty. This scoring makes the confidence number real instead of cosmetic — and the PS explicitly asks for a confidence score per flag.

### 3b. Synthetic dataset generator with KNOWN ground truth (new — needed before 3c can work)
Build a generator that produces synthetic transactions AND a ground-truth label (normal / anomalous) for each entity, by deliberately injecting known patterns:
- Normal: moderate transaction frequency, normal graph connectivity, normal timing
- Anomalous: high velocity, large fan-out (peeling chain), rapid movement through several wallets, amount splitting, layering (long obfuscation chains), circular flow back to origin

This needs to produce enough labeled cases (aim for 200+ transactions, mix of normal and several distinct anomaly types) for 3c to have something to learn from. Keep the existing `sample_transactions.csv` as a small smoke-test file; add this as a separate, larger generated dataset.

### 3c. Risk weights learned, not hand-picked (upgrades whatever risk-combination logic exists)
Once 3b's labeled data exists: compute four evidence signals per entity (AI anomaly score, graph evidence, behavioral evidence, cluster evidence — clustering may not exist yet, stub it as 0 if so), fit a simple logistic regression from those four signals to the known labels, and use the resulting coefficients as the risk-score weights. Do NOT hand-pick weights like 40/20/20/20 — the whole point is that you control the ground truth for synthetic data, so weights should come from fitting, not guessing. Bucket final risk: 0–30 Low, 31–60 Medium, 61–80 High, 81–100 Critical — state clearly in output that these are prototype thresholds, not official NTRO values.

### 3d. Risk ≠ Confidence — keep these two numbers separate, always
- **Risk**: how unusual is this entity's behavior (from AI anomaly + graph + behavioral + cluster evidence)
- **Confidence**: how much do we trust the evidence behind that judgment (from 3a's correlation confidence + how complete the entity's feature data is)

These must be computed from disjoint evidence and are allowed to disagree — that disagreement is itself useful information for an investigator. Every output should show both numbers separately, never collapse them into one.

### 3e. Investigative Lead case file (upgrades the current CLI print output)
Instead of printing `wallet, anomaly_score, reason`, produce a structured record per flagged entity:
```
Entity, Risk score, Confidence score, "Why flagged" (list of specific reasons),
Supporting evidence (specific TXIDs), Related entities, Timeline (ordered events), 
Investigation path (IP -> TXID -> Wallet -> Wallet chain)
```

## 4. Hard constraints — do not violate these

- **Fully offline at runtime.** No live blockchain API calls, no internet dependency once the system is running. Internet is fine for installing packages beforehand, not during operation.
- **Never overclaim.** Say "flags behavior that warrants investigation," never "identifies criminals" or "proves ownership" or "proves laundering." This is a real requirement from the design doc, not a style preference — a judge with domain background will penalize overclaiming hard.
- **One primary ML model.** Isolation Forest is the AI/ML core the PS requires. DBSCAN/graph-clustering/rule-based pattern checks (fan-in, fan-out, rapid movement, splitting, layering, circular flow) are supporting evidence that explains and corroborates the model's verdict — never a second competing verdict. Don't present four outputs that might disagree.
- **Explain, verify, don't assume.** Before considering any new feature or function "done," run it against a case where you already know the right answer (the way the existing fan_out bug was caught) and show the actual output, not just that it executed without error.

## 5. Build order — realistic across 4 remaining nights (Sept 6–9, submit Sept 10)

**TONIGHT (Night 1) — the actual scope for this session:**
1. Section 3b: synthetic dataset generator with labeled ground truth
2. Section 3a: correlation confidence scoring, replacing the current unscored IP-to-tx linking

Stop there tonight. Do not start clustering, the risk engine, or the dashboard yet — those depend on tonight's output existing and working first.

**Night 2:** Feature engineering expanded to the four categories (transaction, network, graph, behavior) from the design doc; Section 3c (learned risk weights) and 3d (risk/confidence split).

**Night 3:** DBSCAN/graph-community clustering; the six pattern-detection heuristics (fan-in, fan-out, rapid movement, splitting, layering, circular flow); Section 3e (investigative lead case file format).

**Night 4:** Minimal Streamlit dashboard (command center + priority alerts list + evidence panel — skip full interactive link-analysis if time is short, a static graph image is acceptable); actual offline test (disconnect internet, confirm it still runs end to end); demo script prep.

**Sept 10:** Submission day. Polish and packaging only, no new features.

## 6. What "done" means for tonight, specifically

Before ending tonight's session, there should be:
- A runnable script that generates the labeled synthetic dataset and prints/saves how many normal vs. how many of each anomaly type were generated
- The correlation confidence scoring function, tested against at least one case with a single unambiguous candidate match (should score high / ACCEPTED) and one case with multiple competing candidates in the same time window (should score lower, possibly UNRESOLVED)
- A plain-language explanation, written for someone reading it tomorrow with no memory of tonight, of exactly what was built, what was tested, and what specifically is NOT done yet
