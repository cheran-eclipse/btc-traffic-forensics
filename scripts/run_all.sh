#!/usr/bin/env bash
# SIH26146 / BitGuard AI -- one-command pipeline run.
#
# Someone who has never seen this repo should be able to run this and watch the
# whole thing work. Everything after `pip install` is offline.
#
#   pip install -r requirements.txt
#   bash scripts/run_all.sh
#
# Add --no-dashboard to skip the (blocking) Streamlit launch at the end.

set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
EXAMPLE_WALLET="1fAJSeQYmUnwYfftUrJsW6dLxX0341"   # a planted circular-flow case

hr() { printf '\n\033[1m======== %s ========\033[0m\n' "$1"; }

hr "1/6  Generate the labelled synthetic dataset (spec 3b)"
"$PY" src/generate_dataset.py --out-dir data --seed 7

hr "2/6  Run the detection pipeline (graph + correlation confidence + Isolation Forest)"
"$PY" src/main.py --input data/synthetic_transactions.csv --top 10

hr "3/6  Per-anomaly-type detection diagnostic"
"$PY" src/diagnostics.py

hr "4/6  DBSCAN clustering + purity report (module 7)"
"$PY" src/clustering.py

hr "5/6  Learned risk weights + Risk/Confidence table (spec 3c / 3d)"
"$PY" src/risk_model.py

hr "6/6  Investigative-lead case file for one planted circular-flow wallet (spec 3e)"
"$PY" src/case_file.py --wallets "$EXAMPLE_WALLET"

if [[ "${1:-}" == "--no-dashboard" ]]; then
  printf '\n\033[1mDone.\033[0m Dashboard skipped. Launch it with:  streamlit run src/dashboard.py\n'
  exit 0
fi

hr "Launching the command-center dashboard (Ctrl+C to stop)"
exec streamlit run src/dashboard.py
