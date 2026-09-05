# Night 1 notes — read this first tomorrow

Session: Sept 6, ~3am. Scope was **only** spec sections 3b and 3a (see
`SIH_BUILD_SPEC.md` section 5, "TONIGHT (Night 1)"). Nothing else was started.

## What was built

### 1. Labelled synthetic dataset generator — `src/generate_dataset.py` (spec 3b)

A script that *invents* Bitcoin-style transaction traffic and, crucially, writes
down the correct answer for every wallet: is it normal, or is it part of a
planted anomaly, and which kind.

- Six anomaly patterns are planted: `high_velocity`, `peeling_chain`,
  `rapid_movement`, `amount_splitting`, `layering`, `circular_flow`. Plus a pile
  of ordinary wallets doing ordinary things.
- Output is two CSV files in `data/`:
  - `synthetic_transactions.csv` — same 12-column schema as the existing
    `data/sample_transactions.csv`, so `src/main.py` reads it unchanged.
  - `synthetic_labels.csv` — one row per wallet: `entity, label, anomaly_type`.
- It is **deterministic**: same `--seed` gives byte-identical output. Default
  seed is 7.
- Default run produces **229 transactions** and **354 labelled wallets**
  (293 normal, 61 anomalous).

Why it exists: sections 3c/3d want the risk-score weights to be *fitted* from
labelled data, not hand-picked (40/20/20/20 etc.). That is only possible if we
own the ground truth. This file is that ground truth. It is a **generator, not a
detector** — it never decides anything is anomalous, it just plants patterns it
already knows the label of.

One honest limitation in the labels: in a peeling chain / hop chain, only the
wallets that actually *send* funds inside the pattern are labelled anomalous.
Wallets that only *receive* a peel or a split-share are left `normal`, because on
their own behaviour they look normal. If tomorrow's feature work wants every
downstream wallet flagged too, that's a labelling-policy change in the
`gen_*` methods.

Run it:

```bash
python src/generate_dataset.py --out-dir data --seed 7
```

### 2. Correlation confidence scoring — `src/correlation_confidence.py` (spec 3a)

`score_correlation(observation, candidates)` takes one network-layer observation
(a packet/flow: a timestamp + a port) and a list of candidate transactions it
*might* correspond to, and returns a confidence number with a status.

The formula is exactly the one in the spec:

```
confidence = 0.5 * time_score + 0.3 * port_score + 0.2 * ambiguity_penalty
```

- `time_score` — 1.0 when the observation and the transaction happen at the same
  moment, 0.5 when they're 60 s apart, trailing to 0 (exponential half-life).
- `port_score` — 1.0 if the observed port is the Bitcoin P2P port (8333), else 0.0.
- `ambiguity_penalty` — `1 / (number of candidate transactions in the same time
  window)`. Despite the name, higher is better: 1 candidate → 1.0, 8 candidates
  → 0.125. This is the whole point of the upgrade: a lone match is trustworthy,
  a match surrounded by 8 equally-close transactions is not.

If confidence ≥ 0.5 the result is `ACCEPTED` and described as an **observation**
("network activity time-correlated with txXXXX — NOT proof of wallet
ownership"). Below 0.5 it's `UNRESOLVED` and left for a human — the code never
forces a guess.

See it run:

```bash
python src/correlation_confidence.py
```

## What was tested (and the actual results)

`python -m pytest` → **19 passed**. Tests are in `tests/`. They are
"known-answer" tests: the input is constructed so the right answer is known.

End-to-end check that matters most — the generator's peeling chains vs. the
*existing, untouched* Isolation Forest in `src/main.py`:

```
planted peeling-chain origin       existing detector says
---------------------------------  ------------------------------------------
1QSJm5AAfT3QyLtNA3mUpf3b            FLAGGED  fan-out=9   "...split into 9 outputs -- peeling-chain pattern"
15tc9GBzKMz5rn7JuRj8nLF8            FLAGGED  fan-out=11  "...split into 11 outputs -- peeling-chain pattern"
1KSRW7pMSm3chqD9tWDqJG8g            FLAGGED  fan-out=10  "...split into 10 outputs -- peeling-chain pattern"
```

All three planted chains were caught, for the right reason, by a detector that
knows nothing about the labels. That's the cross-check.

Correlation confidence on a real generated transaction:

```
scenario                                             status      confidence
---------------------------------------------------  ----------  ----------
true tx is the only candidate, obs 6 s later, :8333  ACCEPTED    0.97
same true tx, but 3 real txs share the time window   ACCEPTED    0.83   (dropped 0.13 from ambiguity alone)
weak obs: ephemeral port 51413, 250 s away           UNRESOLVED  0.35
```

Note: the generated dataset is spread over 5 days, so time windows rarely hold
many competitors — the ambiguity effect is real but modest here. A denser
dataset (or a wider `--instances-per-anomaly`) would show it harder. The unit
tests exercise the crowded-window case directly (8 competitors → 0.78 vs 0.96).

## What is NOT done yet

- **Not wired into `src/main.py`.** `build_graph` still links every IP to every
  transaction with an unweighted edge. `score_correlation` exists and is tested
  as a standalone function; replacing the graph edges with scored ones is the
  first integration task and was deliberately left for later so tonight's PR
  stays small and reviewable.
- Section 3c (logistic-regression-learned risk weights) — needs tonight's
  labelled data, which now exists. This is Night 2.
- Section 3d (risk vs. confidence as two separate numbers) — Night 2.
- Sections 3e / clustering / pattern heuristics / dashboard — Nights 3–4.
- The `accept_threshold` (0.5), `time_halflife_s` (60), and `ambiguity_window_s`
  (600) in `correlation_confidence.py` are prototype defaults, not tuned values.
- Amounts in the generator don't perfectly conserve value across a chain (a
  flat ~1–2% "fee" is skimmed each hop); fine for pattern-shape learning, not
  for balance auditing.

## Files added this session

```
src/generate_dataset.py            generator
src/correlation_confidence.py      scoring function
tests/test_generate_dataset.py     10 tests
tests/test_correlation_confidence.py  9 tests
data/synthetic_transactions.csv    generated (seed 7), committed for reproducibility
data/synthetic_labels.csv          generated (seed 7)
requirements-dev.txt               adds pytest on top of requirements.txt
README.md                          new "BitGuard AI refinements" section
```
