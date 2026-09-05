# Night 4 notes — read this first tomorrow

Session continues from Night 3. Branch: `feat/night4-dashboard-offline-demo`
(cut from the Night 3 branch; **nothing merged to main**). Scope: dashboard,
offline test, demo script. **No detection work** — the remaining recall gaps
were deliberately left as documented limitations (budget: ~25% of the week
left, Sept 10 close, and a judge sees the demo, not the recall curve).

Commits:

```
Add minimal Streamlit command-center dashboard (spec section 19)
Add offline self-check (spec section 4: fully offline at runtime)
Add Night 4 demo script + notes   <- this file
```

---

## 1. Dashboard — `src/dashboard.py`

`streamlit run src/dashboard.py`. Deliberately small, three panels:

- **Command Center** — Transactions 229 · Entities 354 · Model-flagged 81 ·
  Behaviour clusters 8 · High/Critical risk 20.
- **Priority Alerts** — the ranked lead table straight from
  `risk_model.score_entities` (entity, risk, severity, confidence, is_flagged,
  ground-truth). "Show only High/Critical" toggle on by default.
- **Investigative Lead** — a dropdown of leads; picking one renders the full
  `case_file.format_case_file` text next to the static entity-graph PNG and the
  DBSCAN purity table.

No interactive link-analysis graph — it reuses `main.save_graph_snapshot`'s PNG
(spec explicitly allows a static image when time is short). Everything is
`@st.cache_data` / `@st.cache_resource`, so the first load runs the pipeline
once (~30 s) and every rerun after that is instant.

`risk_model.score_entities` now also returns `is_flagged` and `anomaly_score`
per entity, so the dashboard needs a single pipeline pass instead of three.

`.streamlit/config.toml` disables Streamlit's usage-stats telemetry so the
dashboard makes no network calls either.

**Tested** with `streamlit.testing.v1.AppTest` (`tests/test_dashboard.py`): the
script runs with no exception, the five Command Center metrics populate, and
selecting a lead renders a case file. Not a pixel test — a "does it run and
show the right things" test.

---

## 2. Offline self-check — `scripts/offline_selfcheck.py`

`python scripts/offline_selfcheck.py`

Installs a guard that makes any non-loopback `socket.connect` / `connect_ex` /
`getaddrinfo` raise `OfflineViolation`, then runs the whole chain. Actual
output:

```
network guard installed: all non-loopback sockets will raise

[ ok ] generate labelled synthetic dataset (3b)
[ ok ] write dataset CSVs
[ ok ] load + parse transactions
[ ok ] build correlation graph + score network edges (3a)
[ ok ] extract wallet features
[ ok ] Isolation Forest anomaly model
[ ok ] DBSCAN clustering + purity (module 7)
[ ok ] logistic-regression risk weights + risk/confidence (3c/3d)
[ ok ] investigative lead case file (3e) for 1jR7gom6Z4c9HWbr
[ ok ] static entity graph PNG

ALL 10 STAGES RAN OFFLINE -- no outbound connection attempted.
learned risk weights: ai 0.24, graph 0.42, behavioral 0.11, cluster 0.23
```

This is stronger evidence than physically pulling the cable because it names
the exact stage if anything ever reaches for the network. `tests/test_offline_selfcheck.py`
runs it in a subprocess and asserts exit 0 + the "ALL 10 STAGES" line.

Belt-and-braces manual check (also documented in DEMO.md): disconnect the
network and re-run any `src/*.py` entry point or the dashboard — all still work.

---

## 3. Demo script — `DEMO.md`

~5 minutes, walks **one** entity end to end: planted circular-flow wallet
`1fAJSeQYmUnwYfftUrJsW6dLxX0341`.

The arc: four raw metadata rows (funds leave the wallet 17:15 in DE, hop
NL→IN→GB, return 17:51 from US — 36 minutes, back to origin) → `main.py`
correlates the two layers into a graph and the Isolation Forest flags it →
`risk_model.py` shows the learned weights and separate risk/confidence →
`clustering.py` corroborates (5 pure-anomalous clusters) → `case_file.py`
produces the finished lead → the dashboard shows it the way a judge would see
it → `offline_selfcheck.py` proves it never touches the internet.

DEMO.md ends with a one-line story and an explicit "honest limitations" list to
say before a judge asks.

---

## What is tested

`python -m pytest` → **60 passed** (was 55 after Night 3). New:
`tests/test_dashboard.py` (4), `tests/test_offline_selfcheck.py` (1). Suite is
~80 s now — the dashboard + offline tests each run the full pipeline. A
session-scoped shared fixture is the obvious speed-up if it starts to hurt.

---

## What is NOT done (final prototype state)

**Detection (deliberately not touched Night 4 — documented, not fixed):**
- `layering` recall by the raw `is_flagged` is ~0.35. The `linear_chain_length`
  feature separates layering wallets cleanly (~9 vs normal ~1); DBSCAN groups
  them (cluster #1, 20 layering wallets); the risk model weights graph+cluster
  evidence heavily so their *risk score* is high. The Isolation Forest itself
  just under-flags long-chain interior wallets — a single-strong-axis blind
  spot. If picked up later: change the diagnostic to measure `risk_fitted >
  threshold`, or add a rule-assist.
- The Isolation Forest flags ~14% of normal wallets as anomalous. None reach
  High/Critical risk (verified Night 5: highest normal risk is ~59 = Medium),
  so they are Medium-severity noise, not false leads -- but the raw flag rate
  is higher than it should be. (An earlier draft of this note said ~2 normal
  wallets "score High" on accidental length-8 chains; that was stale -- the
  Night 3 cluster-evidence wiring changed the risk scaling and the top normal
  wallets now have chain length 0. The fix would still be a
  `linear_chain_length` cap or a monotonic-amount check if the flag rate is
  tightened later.)
- `behavioral_evidence` has a negative fitted coefficient in 3c (honest output
  of fitting; collinear with graph, and `high_velocity` is only 3 wallets).

**Not built at all:**
- Real NTRO dataset swapped in for the synthetic one.
- GeoIP enrichment (still reads geo columns from the data).
- Any tuning of thresholds against real/held-out data (risk buckets 30/60/80,
  DBSCAN eps 1.4, correlation accept 0.5, six heuristic thresholds — all
  prototype).
- Packaging / install script for a clean machine.

**For Sept 10 (submission day, polish only):**
- Regenerate `output/entity_graph.png` for the README if the demo dataset
  changed.
- One dry run of DEMO.md start to finish on the actual demo machine.
- `pip freeze > requirements-lock.txt` on the demo machine so the offline
  install is reproducible.

## Files this session

```
src/dashboard.py                new — Streamlit command center
src/risk_model.py               score_entities returns is_flagged / anomaly_score
scripts/offline_selfcheck.py    new — socket-guarded full-pipeline run
.streamlit/config.toml          new — disable telemetry
requirements.txt                + streamlit
DEMO.md                         new — 5-minute walk-through
tests/test_dashboard.py         new — 4
tests/test_offline_selfcheck.py new — 1
README.md                       Night 4 status
```
