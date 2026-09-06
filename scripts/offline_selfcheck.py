"""
SIH26146 / BitGuard AI -- offline self-check.

Hard requirement (spec section 4): "Fully offline at runtime. No live
blockchain API calls, no internet dependency once the system is running."

This script proves it without needing to physically pull the network cable:
it installs a guard that makes ANY outbound socket connection raise
OfflineViolation, then runs the whole pipeline end to end --
generate_dataset -> build_graph -> features -> Isolation Forest -> correlation
confidence -> DBSCAN -> logistic-regression risk weights -> case file ->
static graph PNG. If any step reaches for the internet, this script fails
loudly and names the step.

Run:
    python scripts/offline_selfcheck.py

For a belt-and-braces manual check, also literally disconnect the network and
run:  python scripts/offline_selfcheck.py   (or any of the src/*.py entry points)
"""

from __future__ import annotations

import socket
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


class OfflineViolation(RuntimeError):
    pass


_ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _install_network_guard() -> None:
    """Block every outbound connection except loopback."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _check(address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _ALLOWED_HOSTS:
            raise OfflineViolation(
                f"pipeline attempted an outbound network connection to {address!r} "
                f"-- this must run fully offline"
            )

    def guarded_connect(self, address):
        _check(address)
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        _check(address)
        return real_connect_ex(self, address)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex

    def _blocked(*_a, **_k):
        raise OfflineViolation("pipeline attempted a DNS lookup -- must run offline")

    socket.getaddrinfo = _blocked  # type: ignore[assignment]


def main() -> int:
    _install_network_guard()
    print("network guard installed: all non-loopback sockets will raise\n")

    import generate_dataset
    import main as pipeline
    import risk_model
    import clustering
    import case_file

    tmp = Path(tempfile.mkdtemp(prefix="bitguard_offline_"))
    tx_csv = tmp / "synthetic_transactions.csv"
    labels_csv = tmp / "synthetic_labels.csv"

    steps = []

    def step(name, fn):
        try:
            out = fn()
        except OfflineViolation:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            raise
        steps.append(name)
        print(f"[ ok ] {name}")
        return out

    ds = step("generate labelled synthetic dataset (3b) + GeoIP enrichment",
              lambda: generate_dataset.build_dataset(seed=7))
    step("write dataset CSVs", lambda: (
        generate_dataset.write_transactions_csv(ds["transactions"], tx_csv),
        generate_dataset.write_labels_csv(ds["labels"], labels_csv),
    ))
    df = step("load + parse transactions", lambda: pipeline.load_transactions(str(tx_csv)))
    g = step("build correlation graph + score network edges (3a)",
             lambda: pipeline.build_graph(df))
    feats = step("extract wallet features", lambda: pipeline.compute_wallet_features(df, g))
    step("Isolation Forest anomaly model", lambda: pipeline.flag_anomalies(feats))
    step("DBSCAN clustering + purity (module 7)",
         lambda: clustering.run(str(tx_csv), str(labels_csv)))
    table, fit = step("logistic-regression risk weights + risk/confidence (3c/3d)",
                      lambda: risk_model.score_entities(str(tx_csv), str(labels_csv)))
    top = table.iloc[0]["wallet"]
    step(f"investigative lead case file (3e) for {top[:16]}",
         lambda: case_file.generate(str(tx_csv), str(labels_csv), wallets=[top]))
    step("static entity graph PNG", lambda: pipeline.save_graph_snapshot(
        g, set(table[table["is_flagged"].fillna(False)]["wallet"]), str(tmp / "graph.png")))

    print(f"\nALL {len(steps)} STAGES RAN OFFLINE -- no outbound connection attempted.")
    print(f"learned risk weights: "
          + ", ".join(f"{k} {v:.2f}" for k, v in fit["normalised_weights"].items()))
    print(f"artifacts written under: {tmp}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OfflineViolation as v:
        print(f"\n[OFFLINE VIOLATION] {v}")
        sys.exit(1)
