# Night 2 notes — read this first tomorrow

Session: continues from Night 1. Branch: `feat/night2-diagnostics-features-3c-3d`
(cut from the Night 1 branch; **nothing merged to main**). Scope was a fixed
5-item list: diagnostic → features → wire correlation confidence → 3c → 3d.
Clustering, pattern heuristics, the case-file format and the dashboard were
**not** started (Night 3/4).

Commits are small and sequential so any one step can be reverted alone:

```
Track SIH_BUILD_SPEC.md in the repo
Add per-anomaly-type detection diagnostic
Add three multi-hop wallet features for the weak anomaly types
Wire correlation confidence into build_graph's network edges
Add learned risk weights (3c) and separate risk/confidence (3d)
Add Night 2 plain-language notes   <- this file
```

## Step 0 — the diagnostic (before any code changed)

`src/diagnostics.py` runs the **unchanged** pipeline against the Night 1 dataset
(`data/synthetic_transactions.csv`, seed 7) and reports, per anomaly type, the
fraction of that type's labelled-anomalous wallets that get `is_flagged == True`.

```
anomaly type        before (6 features)   after (9 features)
------------------  -------------------   ------------------
amount_splitting          1.00                  1.00
high_velocity             1.00                  1.00
peeling_chain             1.00                  1.00
circular_flow             0.21                  0.86     <- big improvement
layering                  0.26                  0.35     <- small improvement
rapid_movement            0.30                  0.30     <- unchanged (see below)
normal FP rate            0.15                  0.14
```

Prediction going in was "circular_flow, amount_splitting, layering will be weak".
Actual weak set was **circular_flow, layering, rapid_movement** — `amount_splitting`
was already fine (its subject wallet feeds many-output transactions, which the
existing `max_tx_output_fanout` catches).

## Step 1 — three new features in `compute_wallet_features` (`src/main.py`)

One feature per weak pattern, each named for what it measures, each with a
comment on why the original six (all *local* — a wallet's own degree, its own
timing span) were blind to it:

| feature | pattern | what it measures |
|---|---|---|
| `min_return_cycle_hops` | circular_flow | fewest wallet-hops for value leaving a wallet to come back to it (0 = never). A ring A→B→C→A is a path property; every wallet in it looks locally ordinary. |
| `linear_chain_length` | layering | length of the run of consecutive 1-in/1-out "pass-through" wallets through this one. Each hop wallet has fan_in == fan_out == 1 → reads as normal locally; the chain is only visible by walking it. |
| `min_receive_to_forward_minutes` | rapid_movement | shortest gap between receiving value and forwarding it. `time_span_minutes` measures *total* activity span, which hides a mule that receives and forwards within a minute. Capped at 1440. |

`FEATURE_COLS` grew 6 → 9, so the Isolation Forest now sees a 9-dim vector.
Existing columns are unchanged (additive). `_explain_row` got matching
plain-language reasons ("funds return to this wallet after N hops", etc.).

**Why rapid_movement's recall didn't move even though the feature works:** the
feature *is* discriminative — rapid_movement wallets have `linear_chain_length`
4–5 and `min_receive_to_forward_minutes` ≈ 2 min, versus normal wallets at 1 and
1440. But neither value alone is rare (some normal wallets accidentally sit on
length-4 chains); only the *conjunction* is rare. Isolation Forest isolates on
single axes well and conjunctions poorly, and 8 of the 9 axes look normal for a
rapid-movement hop wallet. The logistic regression in step 3 can weight the
signal directly, which is the better home for it.

## Step 2 — correlation confidence wired into `build_graph`

Every `ip -> tx` edge (`kind == "network_link"`) now carries:
- `confidence` — 0–1 from `score_correlation`
- `correlation_status` — `ACCEPTED` or `UNRESOLVED`

The observation is the flow's timestamp + its Bitcoin-side port; the candidates
are every transaction within ±600 s. `score_correlation` got one new optional
arg, `target_txid`, so it scores the *asserted* pairing (the row's own tx) while
still taking the ambiguity penalty from the full candidate list.

Actual result on the Night 1 dataset: **all 458 network edges land ACCEPTED,
confidence 0.81–1.00** (mean 0.92). That's expected — in this dataset the packet
timestamp and the tx timestamp are identical (time_score = 1.0) and the port is
always 8333. A dataset with observation jitter (packet seen seconds/minutes off
the tx, or on an ephemeral port) would produce `UNRESOLVED` edges; the unit
tests exercise that path directly.

`main.py` still runs end to end. The `sample_transactions.csv` demo output now
shows the new reason strings (circular-flow / rapid-movement lines), because the
new features fire on that 20-row file too.

## Step 3 — learned risk weights (spec 3c), `src/risk_model.py`

Four evidence signals per entity, each z-scored so larger = more suspicious:

- `ai_evidence` — Isolation Forest anomaly score
- `graph_evidence` — mean of z-scored fan_in, fan_out, max_tx_output_fanout,
  min_return_cycle_hops, linear_chain_length
- `behavioral_evidence` — mean of z-scored num_tx_touched, distinct_countries,
  −min_receive_to_forward_minutes
- `cluster_evidence` — **0.0 stub** (clustering is Night 3)

A `LogisticRegression` (class-weight balanced, 354 entities, 61 anomalous) is fit
signals → `label == anomalous`. Its coefficients are the risk weights:

```
signal        raw coef   |w| fitted   naive
ai              1.438        0.51       0.40
graph           1.153        0.41       0.20
behavioral      0.227        0.08       0.20
cluster         0.000        0.00       0.20     (zero-variance stub)
intercept      -1.039
```

Fitting moved weight toward **graph evidence** (0.20 → 0.41) and away from
behavioral and cluster. Both `risk_fitted` and `risk_naive` are the same four
min-max-scaled signals weighted and rescaled to 0–100 — only the weights differ —
so the rank changes are purely the effect of fitting:

```
wallet            type            risk_fit   risk_naive   rank change
1KTquEY7qLeuZuVw  high_velocity      89.7       100.0         -3
1zGdNkd247ziKwJd  high_velocity      83.7        92.5         -3
1QcrGrT3u7bYhihg  circular_flow      73.3        63.6         +9
1GM3qzZCbBFmn9Dv  peeling_chain      70.5        66.5         +3
```

The naive split over-weights the raw AI score, so it pushes the high-velocity
wallets (which the IF loves) to the very top and buries circular-flow. Fitting
pulls circular-flow up nine places.

Run it: `python src/risk_model.py`

## Step 4 — Risk and Confidence kept separate (spec 3d)

`score_entities()` returns a table with **both** numbers as separate columns,
never combined:

- **Risk** (0–100, bucketed Low/Medium/High/Critical — prototype thresholds)
  from the four evidence signals above.
- **Confidence** (0–1) from disjoint evidence: `0.6 * mean correlation
  confidence of the entity's network edges + 0.4 * feature completeness`.

They are allowed to disagree, e.g.:

```
wallet            type            RISK   bucket     CONF   (corr / completeness)
1KTquEY7qLeuZuVw  high_velocity   89.7   Critical   0.76   (0.81 / 0.67)
```

High risk, only moderate confidence — the investigator is told "this looks bad
but the evidence is partial", which is the point of not collapsing them.

## What is tested

`python -m pytest` → **35 passed** (was 19 after Night 1). New files:
`tests/test_night2_features.py` (9), `tests/test_risk_model.py` (7).

Known-answer style: hand-built 3-wallet rings / chains for the features; the full
Night 1 dataset for the risk model (learned weights differ from naive by >0.1;
cluster stub carries ~0 weight; fitting changes the ranking; risk and confidence
are not the same number rescaled; anomalous entities average higher risk).

End-to-end checks that ran green:
- `python src/main.py --input data/sample_transactions.csv` — runs, new reasons appear.
- `python src/diagnostics.py` — the before/after table above.
- `python src/risk_model.py` — weights + comparison + risk/confidence table.

## What is NOT done / known problems

1. **Several normal-labelled wallets score high risk.** Two causes: (a) Isolation
   Forest false positives (14% of normals) that the risk model faithfully
   echoes; (b) **label noise from Night 1** — layering and rapid-movement
   *terminal* wallets (the ones that only receive at the end of a chain) were
   labelled `normal` because only senders were labelled, but structurally they
   sit on a long chain and the new `linear_chain_length` feature flags them.
   This is the labelling-policy question flagged in NIGHT1_NOTES. Deciding it is
   a Night 3 task — either relabel chain-terminal wallets, or accept them as
   legitimately-not-of-interest and tune the model.
2. **rapid_movement recall is still 0.30.** The feature is there and
   discriminative; the Isolation Forest just doesn't exploit a single-axis
   conjunction. The 3c logistic regression is where this signal should pay off
   once its weights are trusted.
3. **All network edges are ACCEPTED** on this dataset — the UNRESOLVED path is
   only exercised by unit tests, because the synthetic packets and transactions
   share timestamps. Consider adding timestamp jitter to the Night 1 generator.
4. **`risk_model.py` re-runs the whole pipeline** (build_graph → features →
   flag_anomalies) internally. Fine for 229 rows; will need caching for bulk data.
5. Thresholds still unprincipled: correlation `accept_threshold` 0.5, risk
   buckets 30/60/80, confidence blend 0.6/0.4.
6. Not started: DBSCAN/community clustering, the six pattern heuristics, the
   investigative-lead case-file format (3e), the Streamlit dashboard.

## Files touched this session

```
SIH_BUILD_SPEC.md               now tracked in git
src/main.py                     +3 features, FEATURE_COLS 6->9, build_graph edges scored, new import
src/correlation_confidence.py   +target_txid arg
src/diagnostics.py              new — per-pattern recall
src/risk_model.py               new — 3c + 3d
tests/test_night2_features.py   new — 9 tests
tests/test_risk_model.py        new — 7 tests
README.md                       Night 2 status section
```
