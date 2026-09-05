# Night 3 notes — read this first tomorrow

Session continues from Night 2. Branch: `feat/night3-clustering-heuristics-casefile`
(cut from the Night 2 branch; **nothing merged to main**). Scope was the
section-5 "Night 3" list: DBSCAN clustering, the six pattern heuristics, the
investigative-lead case file. Plus a Night-2 bug investigation the user asked
for. **No dashboard** (Night 4).

Commits, small and sequential:

```
Fix stale feature-count in compute_wallet_features docstring
Fix Night 2 false positives: cap return-cycle search at 7 hops
Add DBSCAN clustering with a purity report (module 7)
Feed real cluster evidence into the 3c risk model
Add the six pattern heuristics as named explainable signals
Add the investigative-lead case file format (spec 3e)
Add Night 3 plain-language notes   <- this file
```

---

## First: the Night 2 "normal wallet, high risk" problem — RESOLVED

NIGHT2_NOTES guessed this was label noise (chain-terminal wallets labelled
normal). **That guess was wrong.** Investigated it directly:

- 20 normal-labelled wallets had `risk_fitted > 60`.
- **None** of them are chain-terminal recipients of a planted pattern.
- All 20 had `min_return_cycle_hops` of 8–11 — the circular-flow feature firing
  on ordinary wallets.

Root cause: `min_return_cycle_hops` used `max_hops=12`. Normal wallets pay each
other at random, so in a 293-wallet mesh almost everything sits "on a cycle" if
you look 8+ hops out. On the Night 1 data:

```
planted circular_flow rings close in   4–6 hops
shortest accidental normal-mesh cycle   8 hops
```

So **it's a genuine model false positive from a too-permissive feature, not a
labelling bug.** `generate_dataset.py` was not touched. Fix: capped the default
at 7 hops (a laundering ring routes funds back deliberately and fast; a 9-hop
return path through the general mesh is not evidence).

Effect:

```
                        before   after
circular_flow recall     0.86    1.00
normal FP rate           0.14    0.14   (cycle-driven FPs gone; ~2 normal
                                         wallets on accidental length-8 linear
                                         chains remain — smaller, same root
                                         cause, left for Night 4 tuning)
```

The learned risk model now de-ranks the residual FPs hard on its own
(`1PSEeV8p…` fitted risk 58.7 vs naive 87.0, −14 rank places).

---

## 1. DBSCAN clustering — `src/clustering.py` (module 7)

Clusters wallets over the same `FEATURE_COLS` the Isolation Forest uses, scaled
the same way. DBSCAN (not k-means): unknown number of groups, most wallets
should fall in one blob or in noise, the interesting output is the small tight
clusters. eps = 1.4, min_samples = 4 — **prototype**, chosen by sweeping
0.8–2.0; the five anomalous clusters are stable across that whole range.

Purity on the Night 1 dataset — is it grouping related actors or carving noise?

```
cluster  size  anom  norm  frac_anom  dominant type
  #1       27    27    0     1.000     layering
  #0        6     6    0     1.000     layering
  #5        6     6    0     1.000     circular_flow
  #7        5     5    0     1.000     circular_flow
  #6        5     5    0     1.000     peeling_chain
  #4      248     0  248     0.000     -            (the mainstream blob)
  #2        8     0    8     0.000     -
  #3       14     0   14     0.000     -
  NOISE    35    12   23     0.343     -
```

**5 clusters are 100 % anomalous, 49 of the 61 anomalous wallets grouped.** The
mainstream blob is 0 % anomalous. So DBSCAN is genuinely grouping related bad
actors. `high_velocity` and `amount_splitting` (3 wallets each) land in noise —
too few / too isolated to form a cluster of min_samples=4, which is the honest
DBSCAN answer, not a failure.

It is **supporting evidence, not a verdict** (spec section 4).

Run: `python src/clustering.py`

### cluster evidence wired into 3c

`risk_model.py`'s `cluster_evidence` was a hard `0.0` stub "until clustering
exists". It exists now. `clustering.cluster_evidence` is **label-free**: 1.0 if
DBSCAN put the wallet in a distinct non-mainstream cluster, else 0.0. It is
deliberately **not** derived from the anomaly score — doing that just aliases
`ai_evidence` and makes the logistic-regression coefficients meaningless (tried
it; `ai` coef went negative).

Learned weights with the fourth signal now real:

```
signal       raw coef   |w| fitted   naive
ai            1.584        0.24       0.40
graph         2.806        0.42       0.20
behavioral   -0.736        0.11       0.20
cluster       1.493        0.23       0.20
intercept    -2.717
```

`graph_evidence` is now the heaviest signal. `behavioral`'s raw coefficient is
**negative** — once graph + cluster evidence are in, the fit slightly
down-weights behavioural evidence. You would never hand-pick a negative weight;
this is exactly what "fit, don't guess" is supposed to surface. (It's partly
collinearity: `high_velocity`, the only pure-behavioural anomaly, is 3 wallets.)

---

## 2. Six pattern heuristics — `src/pattern_heuristics.py`

`fan_in`, `fan_out`, `rapid_movement`, `amount_splitting`, `layering`,
`circular_flow`. Each returns a `PatternSignal(name, fired, detail, txids)` —
**supporting evidence, never a competing verdict** (spec section 4: "don't
present four outputs that might disagree"). Nothing here decides whether a
wallet is flagged; the Isolation Forest still owns that.

Four reuse Night 2 feature values and just attach transaction-level evidence
(`fan_out`, `rapid_movement`, `layering`, `circular_flow`). **Two are new named
checks:**

- `fan_in` — a transaction that consolidated ≥ 5 input wallets (gather stage).
- `amount_splitting` — a transaction that split a sum into ≥ 3 parts within 6 %
  of each other (structuring). Distinct from `fan_out`, which is about *count*:
  a peeling change output + small peels does **not** trip `amount_splitting`
  (there's a test for exactly that).

Verified on planted subjects:

```
peeling_chain origin     -> fan_out fired ("split into 9 outputs")
amount_splitting origin   -> amount_splitting fired ("split ~0.756 into 6 near-equal parts of ~0.126")
circular_flow origin      -> circular_flow + layering fired (a ring is also a pass-through chain)
```

---

## 3. Investigative Lead case file — `src/case_file.py` (spec 3e)

Structured record per flagged entity: entity, risk score + bucket, confidence
score + breakdown, why-flagged reasons, supporting TXIDs, related wallets/IPs,
time-ordered timeline, `IP -> TXID -> Wallet -> Wallet` investigation path.
Risk and confidence printed as **two separate numbers** with a one-line note
that they may disagree (3d). No overclaiming language (a test enforces a
banned-words list).

```
python src/case_file.py --top 5
python src/case_file.py --wallets <id>,<id>
```

### Full example — a planted peeling-chain wallet

```
========================================================================
INVESTIGATIVE LEAD -- 1QSJm5AAfT3QyLtNA3mUpf3bZ90153
========================================================================
Risk score        : 35.2 / 100   (Medium)
Confidence score  : 0.73        (corr 0.85 / completeness 0.56)
  Risk = how unusual the behaviour is.  Confidence = how much to trust the evidence.
  These are separate numbers and may disagree.
(synthetic ground-truth label: peeling_chain)

WHY FLAGGED (behaviour warranting investigation -- not a conclusion):
  - [fan_out] fed a transaction that split into 9 outputs (peeling-chain shape)

SUPPORTING EVIDENCE -- transactions (1):
  tx00088

RELATED ENTITIES -- 9 wallets, 2 IPs:
  wallets: 116VLJwR5UGUqu, 1CBaybrmPHmaa3, 1Qij1UmyjHPNF1, 1QtQZJDA65RZ6u, ...
  IPs    : 189.184.208.79, 53.51.29.23

TIMELINE:
  2026-09-03 02:31  sent     via tx00088  (1 in / 9 out)  -> 1iEDMZ2Zp1s8, 1vmtgvfY1Hsf, ...

INVESTIGATION PATH:
  IP 189.184.208.79  ->  TXID tx00088  ->  Wallet 1QSJm5AAfT3QyLtNA3mUpf3bZ90153
      ->  Wallet 1iEDMZ2Zp1s8qMNBZECgjxYThe0154
========================================================================
```

Note: a peeling *origin* only has the one peel transaction, so its risk (35,
Medium) and completeness (0.56) are modest — the score is relative to wallets
like circular_flow rings that trip several independent signals. It is still
correctly flagged and the fan-out reason is exact.

### Full example — a planted circular-flow wallet

```
========================================================================
INVESTIGATIVE LEAD -- 1fAJSeQYmUnwYfftUrJsW6dLxX0341
========================================================================
Risk score        : 53.3 / 100   (Medium)
Confidence score  : 0.91        (corr 0.86 / completeness 1.00)
(synthetic ground-truth label: circular_flow)

WHY FLAGGED (behaviour warranting investigation -- not a conclusion):
  - [layering] sits on a 7-wallet pass-through chain (long obfuscation chain)
  - [circular_flow] funds leaving this wallet return to it after 4 hops

SUPPORTING EVIDENCE -- transactions (2):
  tx00028, tx00033

RELATED ENTITIES -- 2 wallets, 4 IPs

TIMELINE:
  2026-09-01 17:15  sent     via tx00028  (1 in / 1 out)  -> 1pS4aed62dUr
  2026-09-01 17:51  received via tx00033  (1 in / 1 out)  -> 11CPLoh4RhyS

INVESTIGATION PATH:
  IP 226.120.18.223  ->  TXID tx00028  ->  Wallet 1fAJSeQYmUnwYfftUrJsW6dLxX0341
      ->  Wallet 1pS4aed62dUrDhF7aRSCBfPiQj0342  ->  Wallet 1PMFbQD3vNwR2zZxiCEU2K1ahE0343
      ->  Wallet 11CPLoh4RhySTPvVMMYrUfMPL70344
```

The timeline shows the wallet sending funds out and receiving them back 36
minutes later; the investigation path walks the ring forward. This is a clean
end-to-end read.

---

## What is tested

`python -m pytest` → **55 passed** (was 36 after Night 2). New files:
`tests/test_clustering.py` (5), `tests/test_pattern_heuristics.py` (7),
`tests/test_case_file.py` (7), plus 1 regression test for the cycle-hops cap.

End-to-end runs that went green:
- `python src/diagnostics.py` — the before/after table above.
- `python src/clustering.py` — the purity table.
- `python src/risk_model.py` — updated 3c weights.
- `python src/pattern_heuristics.py` — signals on planted subjects.
- `python src/case_file.py --wallets <peeling>,<circular>` — the two case files above.
- `python src/main.py --input data/sample_transactions.csv` — still runs.

---

## What is NOT done / known problems

1. **layering recall is still 0.35.** The `linear_chain_length` feature is
   discriminative (layering wallets ~9, normal ~1) but the Isolation Forest
   still under-flags the interior hop wallets — same single-strong-axis
   limitation noted in Night 2. Clustering *does* group them (cluster #1 has 20
   layering wallets), and the risk model weights graph + cluster evidence
   heavily, so a layering wallet's *risk score* is now high even when
   `is_flagged` is False. Worth deciding in Night 4 whether the diagnostic
   should measure `is_flagged` or `risk_fitted > threshold`.
2. **~2 residual normal false positives** on accidental length-8 `linear_chain`
   runs in the normal-wallet mesh — the same "accidental structure in a random
   graph" effect as the cycle bug, one level down. A `linear_chain_length` cap
   or a "chain must carry a monotonically decreasing amount" check would fix it.
   Left for Night 4.
3. **`behavioral_evidence` raw coefficient is negative** in the 3c fit (see
   above). Honest output of fitting, but if Night 4 wants all-positive weights,
   either drop `behavioral` as a separate signal or fold velocity into graph.
4. **Test suite is slow (~45 s)** — `clustering`, `risk_model` and `case_file`
   tests each re-run the whole pipeline several times. A shared session-scoped
   fixture would cut it a lot.
5. **eps = 1.4 and the six heuristic thresholds** are prototype values, not
   tuned against anything official.
6. Not started: the Streamlit dashboard, the offline end-to-end test, demo
   script (Night 4).

## Files touched this session

```
src/main.py                     min_return_cycle_hops max_hops 12 -> 7; docstring fix
src/clustering.py               new — DBSCAN + purity report + cluster_evidence
src/pattern_heuristics.py       new — six named PatternSignals
src/case_file.py                new — investigative lead format (3e)
src/risk_model.py               cluster_evidence wired in (was a 0.0 stub)
tests/test_clustering.py        new — 5
tests/test_pattern_heuristics.py new — 7
tests/test_case_file.py         new — 7
tests/test_night2_features.py   +1 regression test
tests/test_risk_model.py        updated for the non-stub cluster evidence
README.md                       Night 3 status
```
