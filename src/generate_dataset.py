"""
SIH26146 / BitGuard AI -- Section 3b: synthetic dataset generator with KNOWN
ground truth.

Why this exists: sections 3c/3d want the risk-score weights to be *learned*
from labelled data instead of hand-picked. That only works if we control the
ground truth. This generator plants known behaviour patterns and records, for
every wallet, whether it is normal or which anomaly type it belongs to.

Nothing in this file is a detector. It only *injects* patterns:

    normal            moderate frequency, normal connectivity, normal timing
    high_velocity     one wallet firing many transactions within minutes
    peeling_chain     one input splitting into many near-equal small outputs,
                      the change output feeding the next such split
    rapid_movement    funds hopping A->B->C->... through several wallets and
                      countries within seconds/minutes each
    amount_splitting  a balance broken into many equal chunks across several
                      transactions (structuring)
    layering          a long obfuscation chain (8-12 hops) over a longer period
    circular_flow     funds routed through a ring of wallets back to the origin

Outputs two CSVs:
    <tx-out>      same schema as data/sample_transactions.csv
                  (timestamp, src_ip, dst_ip, src_port, dst_port, txid,
                   input_addresses, output_addresses, input_amounts,
                   output_amounts, geo_country, asn)
    <labels-out>  entity, label (normal|anomalous), anomaly_type

The existing data/sample_transactions.csv stays as the small smoke-test file;
this is a separate, larger, labelled dataset.

Offline: standard library + numpy only. No network.

Run:
    python src/generate_dataset.py --out-dir data --seed 7
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

BITCOIN_P2P_PORT = 8333
BASE_TIME = datetime(2026, 9, 1, 0, 0, 0)
_MAX_START_OFFSET_MIN = 60 * 24 * 5  # patterns start somewhere across a 5-day window

# (geo_country, asn) pairs drawn from for the network layer.
GEO_POOL: list[tuple[str, str]] = [
    ("IN", "AS9829"), ("NL", "AS60781"), ("DE", "AS3320"), ("US", "AS7922"),
    ("RU", "AS12389"), ("SG", "AS9299"), ("GB", "AS5089"), ("FR", "AS3215"),
]

ANOMALY_TYPES: list[str] = [
    "high_velocity", "peeling_chain", "rapid_movement",
    "amount_splitting", "layering", "circular_flow",
]

_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


class _Generator:
    """Accumulates transactions and per-wallet ground-truth labels."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.txs: list[dict[str, Any]] = []
        self.subject_labels: dict[str, str] = {}  # wallet -> anomaly_type
        self.instances: list[dict[str, Any]] = []  # one record per planted pattern
        self._addr_n = 0

    # -- primitives --------------------------------------------------------

    def addr(self) -> str:
        """A deterministic pseudo-address. Not a real key, not a real hash."""
        self._addr_n += 1
        body = "".join(self.rng.choice(list(_B58), size=25))
        return f"1{body}{self._addr_n:04d}"

    def ip(self) -> str:
        return ".".join(str(int(x)) for x in self.rng.integers(1, 254, size=4))

    def _start_time(self) -> datetime:
        return BASE_TIME + timedelta(
            minutes=int(self.rng.integers(0, _MAX_START_OFFSET_MIN))
        )

    def _emit(
        self,
        ts: datetime,
        inputs: list[str],
        outputs: list[str],
        in_amts: list[float],
        out_amts: list[float],
        geo: tuple[str, str] | None = None,
    ) -> None:
        g, asn = geo if geo is not None else GEO_POOL[
            int(self.rng.integers(0, len(GEO_POOL)))
        ]
        self.txs.append(
            {
                "timestamp": ts,
                "src_ip": self.ip(),
                "dst_ip": self.ip(),
                "src_port": int(self.rng.integers(49152, 65535)),
                "dst_port": BITCOIN_P2P_PORT,
                "txid": None,  # assigned after everything is time-sorted
                "input_addresses": list(inputs),
                "output_addresses": list(outputs),
                "input_amounts": [float(a) for a in in_amts],
                "output_amounts": [float(a) for a in out_amts],
                "geo_country": g,
                "asn": asn,
            }
        )

    def _label(self, wallets: list[str], anomaly_type: str) -> None:
        for w in wallets:
            self.subject_labels[w] = anomaly_type

    def _record_instance(self, anomaly_type: str, origin: str, members: list[str]) -> None:
        self.instances.append(
            {"anomaly_type": anomaly_type, "origin": origin, "members": members}
        )

    # -- pattern injectors ----------------------------------------------------
    #
    # Labelling rule: a wallet is labelled with an anomaly type when it *acts*
    # inside that pattern -- i.e. it is a sender (appears in input_addresses) of
    # a transaction the pattern created. Wallets that only receive a peel / a
    # split share are downstream and stay "normal": on their own behaviour they
    # look normal, which is the honest label for section 3c to learn from.

    def gen_normal(self, n_wallets: int) -> list[str]:
        wallets = [self.addr() for _ in range(n_wallets)]
        for w in wallets:
            t = self._start_time()
            balance = float(self.rng.uniform(0.3, 3.0))
            for _ in range(int(self.rng.integers(2, 5))):
                t = t + timedelta(minutes=int(self.rng.integers(25, 600)))
                counterparty = (
                    self.addr()
                    if self.rng.random() < 0.5
                    else wallets[int(self.rng.integers(0, len(wallets)))]
                )
                send = round(balance * float(self.rng.uniform(0.1, 0.4)), 8)
                send = max(send, 0.01)
                received = round(max(send * (1.0 - 0.01), 0.0), 8)
                self._emit(t, [w], [counterparty], [send], [received])
                balance = max(balance - send, 0.05)
        return wallets

    def gen_high_velocity(self, k: int) -> None:
        for _ in range(k):
            w = self.addr()
            self._label([w], "high_velocity")
            self._record_instance("high_velocity", w, [w])
            t = self._start_time()
            for _ in range(int(self.rng.integers(9, 16))):
                t = t + timedelta(seconds=int(self.rng.integers(8, 55)))
                amt = round(float(self.rng.uniform(0.01, 0.2)), 8)
                self._emit(t, [w], [self.addr()], [amt], [round(amt * 0.99, 8)])

    def gen_peeling_chain(self, k: int) -> None:
        for _ in range(k):
            origin = self.addr()
            t = self._start_time()
            current = origin
            senders = [origin]
            amount = float(self.rng.uniform(1.0, 4.0))
            for _ in range(int(self.rng.integers(2, 4))):
                t = t + timedelta(minutes=int(self.rng.integers(1, 8)))
                n_out = int(self.rng.integers(9, 13))
                peel = round(amount * 0.02, 8)
                recipients = [self.addr() for _ in range(n_out - 1)]
                change = self.addr()
                change_amt = round(max(amount - peel * (n_out - 1), 0.01), 8)
                self._emit(
                    t,
                    [current],
                    recipients + [change],
                    [round(amount, 8)],
                    [peel] * (n_out - 1) + [change_amt],
                )
                current, amount = change, change_amt
                senders.append(change)
            # the last `change` wallet never spends -> it is not a sender
            self._label(senders[:-1], "peeling_chain")
            self._record_instance("peeling_chain", origin, senders[:-1])

    def gen_rapid_movement(self, k: int) -> None:
        for _ in range(k):
            chain = [self.addr() for _ in range(int(self.rng.integers(4, 7)))]
            self._label(chain[:-1], "rapid_movement")
            self._record_instance("rapid_movement", chain[0], chain[:-1])
            t = self._start_time()
            amount = round(float(self.rng.uniform(0.5, 2.0)), 8)
            for a, b in zip(chain, chain[1:]):
                t = t + timedelta(seconds=int(self.rng.integers(30, 180)))
                sent = amount
                amount = round(amount * 0.99, 8)
                self._emit(
                    t, [a], [b], [sent], [amount],
                    geo=GEO_POOL[int(self.rng.integers(0, len(GEO_POOL)))],
                )

    def gen_amount_splitting(self, k: int) -> None:
        for _ in range(k):
            w = self.addr()
            self._label([w], "amount_splitting")
            self._record_instance("amount_splitting", w, [w])
            t = self._start_time()
            total = float(self.rng.uniform(1.0, 6.0))
            n_splits = int(self.rng.integers(3, 6))
            chunk = round(total / n_splits, 8)
            for _ in range(n_splits):
                t = t + timedelta(minutes=int(self.rng.integers(2, 30)))
                parts = int(self.rng.integers(4, 8))
                each = round(chunk / parts, 8)
                self._emit(
                    t, [w], [self.addr() for _ in range(parts)],
                    [round(chunk, 8)], [each] * parts,
                )

    def gen_layering(self, k: int) -> None:
        for _ in range(k):
            chain = [self.addr() for _ in range(int(self.rng.integers(8, 13)))]
            self._label(chain[:-1], "layering")
            self._record_instance("layering", chain[0], chain[:-1])
            t = self._start_time()
            amount = round(float(self.rng.uniform(1.0, 5.0)), 8)
            for a, b in zip(chain, chain[1:]):
                t = t + timedelta(minutes=int(self.rng.integers(10, 90)))
                sent = amount
                amount = round(amount * 0.985, 8)
                self._emit(
                    t, [a], [b], [sent], [amount],
                    geo=GEO_POOL[int(self.rng.integers(0, len(GEO_POOL)))],
                )

    def gen_circular_flow(self, k: int) -> None:
        for _ in range(k):
            ring = [self.addr() for _ in range(int(self.rng.integers(4, 7)))]
            self._label(ring, "circular_flow")  # every ring wallet both sends and receives
            self._record_instance("circular_flow", ring[0], ring)
            t = self._start_time()
            amount = round(float(self.rng.uniform(0.3, 1.5)), 8)
            loop = ring + [ring[0]]  # ... back to the origin
            for a, b in zip(loop, loop[1:]):
                t = t + timedelta(minutes=int(self.rng.integers(2, 20)))
                sent = amount
                amount = round(amount * 0.98, 8)
                self._emit(t, [a], [b], [sent], [amount])

    # -- finalisation -------------------------------------------------------

    def finalise(self) -> None:
        """Time-sort the transactions and assign stable txids."""
        self.txs.sort(key=lambda r: r["timestamp"])
        for i, row in enumerate(self.txs, start=1):
            row["txid"] = f"tx{i:05d}"

    def label_rows(self) -> list[dict[str, str]]:
        seen: set[str] = set()
        for tx in self.txs:
            seen.update(tx["input_addresses"])
            seen.update(tx["output_addresses"])
        rows = []
        for wallet in sorted(seen):
            atype = self.subject_labels.get(wallet)
            rows.append(
                {
                    "entity": wallet,
                    "label": "anomalous" if atype else "normal",
                    "anomaly_type": atype or "none",
                }
            )
        return rows


def build_dataset(
    seed: int = 7,
    n_normal: int = 45,
    instances_per_anomaly: int = 3,
) -> dict[str, Any]:
    """Generate the labelled dataset in memory (no files written).

    Returns a dict with keys:
        transactions  list[dict]  -- rows ready for CSV / for src/main.py
        labels        list[dict]  -- entity, label, anomaly_type
        instances     list[dict]  -- one record per planted pattern
                                     (anomaly_type, origin, members)
        summary       dict        -- counts, for printing and for tests
    """
    gen = _Generator(seed)
    gen.gen_normal(n_normal)
    gen.gen_high_velocity(instances_per_anomaly)
    gen.gen_peeling_chain(instances_per_anomaly)
    gen.gen_rapid_movement(instances_per_anomaly)
    gen.gen_amount_splitting(instances_per_anomaly)
    gen.gen_layering(instances_per_anomaly)
    gen.gen_circular_flow(instances_per_anomaly)
    gen.finalise()

    labels = gen.label_rows()
    by_type: dict[str, int] = {"normal": 0}
    for row in labels:
        key = row["anomaly_type"] if row["label"] == "anomalous" else "normal"
        by_type[key] = by_type.get(key, 0) + 1

    injected: dict[str, int] = {}
    for inst in gen.instances:
        injected[inst["anomaly_type"]] = injected.get(inst["anomaly_type"], 0) + 1

    summary = {
        "seed": seed,
        "n_transactions": len(gen.txs),
        "n_entities": len(labels),
        "n_normal": by_type.get("normal", 0),
        "n_anomalous": sum(v for k, v in by_type.items() if k != "normal"),
        "injected_instances": {k: injected.get(k, 0) for k in ANOMALY_TYPES},
        "by_anomaly_type": {k: by_type.get(k, 0) for k in ANOMALY_TYPES},
    }
    return {
        "transactions": gen.txs,
        "labels": labels,
        "instances": gen.instances,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# CSV serialisation
# ---------------------------------------------------------------------------

_TX_COLUMNS = [
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid",
    "input_addresses", "output_addresses", "input_amounts", "output_amounts",
    "geo_country", "asn",
]


def _fmt_amount(x: float) -> str:
    return f"{x:.8f}"


def write_transactions_csv(transactions: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_TX_COLUMNS)
        for row in transactions:
            writer.writerow(
                [
                    row["timestamp"].isoformat(),
                    row["src_ip"], row["dst_ip"],
                    row["src_port"], row["dst_port"],
                    row["txid"],
                    ";".join(row["input_addresses"]),
                    ";".join(row["output_addresses"]),
                    ";".join(_fmt_amount(a) for a in row["input_amounts"]),
                    ";".join(_fmt_amount(a) for a in row["output_amounts"]),
                    row["geo_country"], row["asn"],
                ]
            )


def write_labels_csv(labels: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["entity", "label", "anomaly_type"])
        writer.writeheader()
        writer.writerows(labels)


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"seed                : {summary['seed']}")
    print(f"transactions        : {summary['n_transactions']}")
    print(f"labelled entities   : {summary['n_entities']}")
    print(f"  normal            : {summary['n_normal']}")
    print(f"  anomalous         : {summary['n_anomalous']}")
    print("    (planted instances / labelled entities, per type)")
    for atype in ANOMALY_TYPES:
        inst = summary["injected_instances"].get(atype, 0)
        ent = summary["by_anomaly_type"].get(atype, 0)
        print(f"    {atype:16s}: {inst} / {ent}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-normal", type=int, default=45)
    ap.add_argument("--instances-per-anomaly", type=int, default=3)
    ap.add_argument("--tx-out", default="synthetic_transactions.csv")
    ap.add_argument("--labels-out", default="synthetic_labels.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = build_dataset(
        seed=args.seed,
        n_normal=args.n_normal,
        instances_per_anomaly=args.instances_per_anomaly,
    )

    tx_path = out_dir / args.tx_out
    labels_path = out_dir / args.labels_out
    write_transactions_csv(result["transactions"], tx_path)
    write_labels_csv(result["labels"], labels_path)

    _print_summary(result["summary"])
    print()
    print(f"transactions -> {tx_path}")
    print(f"labels       -> {labels_path}")

    if result["summary"]["n_transactions"] < 200:
        print("\nWARNING: fewer than 200 transactions; raise --n-normal or "
              "--instances-per-anomaly for section 3c training.")


if __name__ == "__main__":
    main()
