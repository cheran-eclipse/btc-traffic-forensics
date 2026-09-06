"""
SIH26146 / BitGuard AI -- minimal command-center dashboard (spec section 19).

Deliberately small: this is the thing a judge looks at, not a product. Three
panels:

  1. Command Center  -- headline counts + the full monitored graph (scale)
  2. Priority Alerts -- the ranked lead table from risk_model.py
  3. Case File       -- pick a lead, see the full investigative record
                        (case_file.py) plus a focused per-lead subgraph
                        (subgraph.py) showing just that lead's money path

The per-lead subgraph replaces the old full-dataset hairball as the evidence
view; the full-dataset image stays in the Command Center as a "scale of what
is being monitored" visual.

Fully offline: every import is local or a pre-installed package; no network
calls anywhere in the pipeline it drives.

Run:
    streamlit run src/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import case_file
import clustering
import main as pipeline
import risk_model
import subgraph

ROOT = _SRC.parent
DEFAULT_TX = str(ROOT / "data" / "synthetic_transactions.csv")
DEFAULT_LABELS = str(ROOT / "data" / "synthetic_labels.csv")
GRAPH_PNG = str(ROOT / "output" / "entity_graph.png")
LEAD_PNG_DIR = ROOT / "output" / "leads"

st.set_page_config(page_title="BitGuard AI -- Command Center", layout="wide")


@st.cache_data(show_spinner="Running the pipeline...")
def _load(tx_csv: str, labels_csv: str):
    table, fit = risk_model.score_entities(tx_csv, labels_csv)
    clusters, purity, _ = clustering.run(tx_csv, labels_csv)
    df = pipeline.load_transactions(tx_csv)
    return table, fit, clusters, purity, len(df)


@st.cache_data(show_spinner="Building case file...")
def _case(tx_csv: str, labels_csv: str, wallet: str):
    files = case_file.generate(tx_csv, labels_csv, wallets=[wallet])
    return files[0] if files else None


@st.cache_data(show_spinner="Drawing the lead subgraph...")
def _lead_subgraph_png(tx_csv: str, labels_csv: str, wallet: str) -> str | None:
    cf = _case(tx_csv, labels_csv, wallet)
    if not cf or "subgraph" not in cf:
        return None
    LEAD_PNG_DIR.mkdir(parents=True, exist_ok=True)
    out = str(LEAD_PNG_DIR / f"{wallet[:16]}.png")
    return subgraph.render(cf["subgraph"], out, cf.get("ground_truth_label"))


@st.cache_resource(show_spinner="Rendering entity graph...")
def _ensure_graph(tx_csv: str) -> str | None:
    Path(GRAPH_PNG).parent.mkdir(parents=True, exist_ok=True)
    df = pipeline.load_transactions(tx_csv)
    g = pipeline.build_graph(df)
    ranked = pipeline.flag_anomalies(pipeline.compute_wallet_features(df, g))
    flagged = set(ranked[ranked["is_flagged"]]["wallet"])
    pipeline.save_graph_snapshot(g, flagged, GRAPH_PNG)
    return GRAPH_PNG if Path(GRAPH_PNG).exists() else None


# --- sidebar ---------------------------------------------------------------
st.sidebar.header("Data")
tx_csv = st.sidebar.text_input("transactions CSV", DEFAULT_TX)
labels_csv = st.sidebar.text_input("labels CSV", DEFAULT_LABELS)
st.sidebar.caption("Prototype thresholds throughout -- not official NTRO values. "
                   "Flags behaviour that warrants investigation; never proves ownership.")

table, fit, clusters, purity, n_tx = _load(tx_csv, labels_csv)

n_clusters = int(clusters[clusters["cluster"] >= 0]["cluster"].nunique())
n_flagged = int(table["is_flagged"].fillna(False).sum())
n_high = int(table["risk_bucket"].isin(["High", "Critical"]).sum())

# --- 1. command center ---------------------------------------------------
st.title("BitGuard AI -- Command Center")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transactions", f"{n_tx:,}")
c2.metric("Entities", f"{len(table):,}")
c3.metric("Model-flagged", n_flagged)
c4.metric("Behaviour clusters", n_clusters)
c5.metric("High / Critical risk", n_high)

with st.expander("Full monitored graph — scale of what is being watched", expanded=False):
    full_png = _ensure_graph(tx_csv)
    if full_png:
        st.image(full_png, width="stretch",
                 caption="Every entity in the dataset (red = model-flagged). "
                         "Per-lead evidence is the focused subgraph below.")

# --- 2. priority alerts -------------------------------------------------
st.subheader("Priority Alerts")
st.caption("Ranked by learned risk score. Risk = how unusual the behaviour is; "
           "Confidence = how much to trust the evidence. They are separate and may disagree.")

view = table.copy()
view["risk"] = view["risk_fitted"].round(1)
view["confidence"] = view["confidence"].round(2)
view = view.rename(columns={"risk_bucket": "severity", "anomaly_type": "ground_truth"})
cols = ["wallet", "risk", "severity", "confidence", "is_flagged", "ground_truth"]
only_high = st.checkbox("show only High / Critical", value=True)
shown = view[view["severity"].isin(["High", "Critical"])] if only_high else view
st.dataframe(shown[cols], width="stretch", hide_index=True, height=320)

# --- 3. case file ------------------------------------------------------
st.subheader("Investigative Lead")
options = list(shown["wallet"]) or list(view["wallet"])
wallet = st.selectbox("entity", options, index=0)

cf = _case(tx_csv, labels_csv, wallet)

lead_png = _lead_subgraph_png(tx_csv, labels_csv, wallet)
if lead_png:
    st.markdown("**This lead's money path** — only the wallets, IPs and "
                "transactions connected to this entity")
    st.image(lead_png, width="stretch")

left, right = st.columns([3, 2])
with left:
    if cf:
        st.code(case_file.format_case_file(cf), language="text")
    else:
        st.info("No case file for this entity.")
with right:
    st.markdown("**Behaviour clusters (DBSCAN purity)**")
    st.dataframe(purity, width="stretch", hide_index=True, height=240)

st.caption(f"Learned risk weights (spec 3c): "
           + ", ".join(f"{k} {v:.2f}" for k, v in fit["normalised_weights"].items()))
