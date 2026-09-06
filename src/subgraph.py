"""
SIH26146 / BitGuard AI -- per-lead investigative subgraph.

The dashboard's old "entity graph" panel drew the whole dataset for every lead
-- a force-directed hairball. This builds, per lead, a *small* subgraph that
shows only that lead's money path and renders it so the pattern is readable:

  - nodes: the subject wallet, the wallets and IPs it is actually connected to,
    and the fan-out transactions that split funds. A simple 1-in / 1-out
    transaction is collapsed onto the wallet->wallet edge (its txid, amount and
    relay IP become edge labels) so a hop chain doesn't double its width. A
    wide fan-out keeps a few real recipients plus a "+N recipients" node.
  - layout: left to right along the money flow (BFS depth, which matches
    first-touch time here), so the flow reads chronologically.
  - edges: directed, arrow-headed, amounts labelled. An edge that points
    backward in the flow -- a ring closing -- is a curved red return arc.
  - key nodes carry text labels (SUBJECT, ring start, terminals, every relay
    IP with its country), not just colour.

Reuses the graph objects the pipeline already built (`g`, `wt`) and the
transactions the pattern heuristics already flagged (`supporting_txids`) -- it
does not re-run detection.

Offline: matplotlib (Agg) + networkx, both already dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

MAX_HOPS = 6
MAX_NODES = 40
FAN_CAP = 6           # real recipients drawn per fan-out tx; the rest collapse to one node


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------

def _tx_inputs(g, tx):
    return [u for u, _, e in g.in_edges(tx, data=True) if e.get("kind") == "input"]


def _tx_outputs(g, tx):
    return [v for _, v, e in g.out_edges(tx, data=True) if e.get("kind") == "output"]


def _tx_ips(g, tx):
    return sorted({u for u, _, e in g.in_edges(tx, data=True)
                   if e.get("kind") == "network_link"})


def _edge_amount(g, u, v, kind):
    data = g.get_edge_data(u, v) or {}
    for e in data.values():
        if e.get("kind") == kind and e.get("amount") is not None:
            return round(float(e["amount"]), 8)
    return None


def _connecting_txs(g, a, b):
    if a not in g:
        return []
    return [tx for tx in g.successors(a)
            if g.nodes[tx].get("kind") == "tx" and b in _tx_outputs(g, tx)]


def _ip_geo(g, ip):
    for _, _, e in g.out_edges(ip, data=True):
        if e.get("kind") == "network_link" and e.get("geo"):
            return e["geo"]
    return ""


# ---------------------------------------------------------------------------
# 1. build the spec
# ---------------------------------------------------------------------------

def build_spec(
    g: nx.MultiDiGraph,
    wt: nx.DiGraph,
    entity: str,
    seed_txids: list[str],
    max_hops: int = MAX_HOPS,
    max_nodes: int = MAX_NODES,
) -> dict[str, Any]:
    tx_all: set[str] = {t for t in seed_txids if t in g}

    def walk(forward: bool):
        seen = {entity}
        frontier = [entity]
        for _ in range(max_hops):
            nxt = []
            for u in frontier:
                if u not in wt:
                    continue
                for v in (wt.successors(u) if forward else wt.predecessors(u)):
                    a, b = (u, v) if forward else (v, u)
                    for tx in _connecting_txs(g, a, b):
                        tx_all.add(tx)
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            frontier = nxt

    walk(forward=True)
    walk(forward=False)

    subj_txs = sorted(
        (t for t in tx_all if entity in _tx_inputs(g, t) or entity in _tx_outputs(g, t)),
        key=lambda t: g.nodes[t]["timestamp"],
    )
    if subj_txs:
        anchor = g.nodes[subj_txs[0]]["timestamp"]
        tx_all = set(sorted(
            tx_all, key=lambda t: abs((g.nodes[t]["timestamp"] - anchor).total_seconds()),
        )[:max_nodes])

    spine_senders = {w for t in tx_all for w in _tx_inputs(g, t)}

    nodes: list[dict] = []
    edges: list[dict] = []
    wallets: set[str] = set()
    ip_labels: dict[str, str] = {}
    collapsed_total = 0

    for tx in sorted(tx_all, key=lambda t: g.nodes[t]["timestamp"]):
        ins, outs = _tx_inputs(g, tx), _tx_outputs(g, tx)
        ts = g.nodes[tx]["timestamp"].isoformat()
        ip = (_tx_ips(g, tx)[:1] or [None])[0]
        geo = _ip_geo(g, ip) if ip else ""
        if ip:
            ip_labels[ip] = f"{ip}  {geo}".strip()

        # simple 1->1 transaction: collapse onto a wallet->wallet edge
        if len(ins) == 1 and len(outs) == 1:
            a, b = ins[0], outs[0]
            wallets.update((a, b))
            edges.append({
                "src": a, "dst": b, "kind": "transfer",
                "txid": tx, "ts": ts, "ip": ip, "geo": geo,
                "amount": _edge_amount(g, tx, b, "output"), "spine": True,
            })
            continue

        # fan-out / consolidation: keep the tx as a node
        out_amt = {w: _edge_amount(g, tx, w, "output") for w in outs}
        for w in ins:
            wallets.add(w)
            edges.append({"src": w, "dst": tx, "kind": "input",
                          "amount": _edge_amount(g, w, tx, "input"), "spine": True})
        keep = [w for w in outs if w in spine_senders or w == entity]
        fan = sorted((w for w in outs if w not in keep), key=lambda w: -(out_amt.get(w) or 0))
        for w in keep + fan[:FAN_CAP]:
            wallets.add(w)
            edges.append({"src": tx, "dst": w, "kind": "output",
                          "amount": out_amt.get(w), "spine": w in spine_senders or w == entity})
        if fan[FAN_CAP:]:
            hidden = fan[FAN_CAP:]
            collapsed_total += len(hidden)
            each = out_amt.get(hidden[len(hidden) // 2])
            mid = f"{tx}~more"
            nodes.append({"id": mid, "kind": "more",
                          "label": f"+{len(hidden)} recipients", "each": each})
            edges.append({"src": tx, "dst": mid, "kind": "output", "amount": each, "spine": False})
        if ip:
            edges.append({"src": ip, "dst": tx, "kind": "network_link", "amount": None})
        nodes.append({"id": tx, "kind": "tx", "label": tx, "ts": ts,
                      "n_in": len(ins), "n_out": len(outs)})

    in_w = {e["src"] for e in edges if e["kind"] in ("input", "transfer")}
    out_w = {e["dst"] for e in edges if e["kind"] in ("output", "transfer")}
    ring_start = _ring_start(g, tx_all, wallets)

    for w in wallets:
        if w == entity:
            role = "subject"
        elif w == ring_start:
            role = "origin"
        elif w in out_w and w not in in_w:
            role = "terminal"
        elif w in in_w and w not in out_w:
            role = "origin"
        else:
            role = "intermediary"
        nodes.append({"id": w, "kind": "wallet", "role": role, "label": _wallet_label(w, role)})
    for ip, lbl in ip_labels.items():
        nodes.append({"id": ip, "kind": "ip", "role": "ip", "label": lbl})

    spine_ids = ({e["src"] for e in edges if e["kind"] in ("input", "transfer")}
                 | {e["dst"] for e in edges if e["kind"] == "transfer"}) & wallets

    return {
        "entity": entity,
        "nodes": nodes,
        "edges": edges,
        "spine_ids": sorted(spine_ids),
        "n_tx": len(tx_all),
        "n_wallets": len(wallets) + collapsed_total,
        "n_ips": len(ip_labels),
        "collapsed": collapsed_total,
    }


def _ring_start(g, tx_nodes, wallets):
    for w in wallets:
        sends = [g.nodes[t]["timestamp"] for t in tx_nodes if w in _tx_inputs(g, t)]
        recvs = [g.nodes[t]["timestamp"] for t in tx_nodes if w in _tx_outputs(g, t)]
        if sends and recvs and max(recvs) > min(sends):
            return w
    return None


def _wallet_label(w: str, role: str) -> str:
    tag = {"subject": "   ⚑ SUBJECT", "origin": "   (ring start)",
           "terminal": "   (terminal)"}.get(role, "")
    if role == "intermediary":
        return f"{w[:10]}…"
    return f"{w[:12]}…{tag}"


# ---------------------------------------------------------------------------
# 2. layout
# ---------------------------------------------------------------------------

def choose_layout(spec: dict) -> dict[str, tuple[float, float]]:
    from datetime import datetime

    by_id = {n["id"]: n for n in spec["nodes"]}
    ids = set(by_id)

    ts: dict[str, float] = {}
    for n in spec["nodes"]:
        if n["kind"] == "tx":
            ts[n["id"]] = datetime.fromisoformat(n["ts"]).timestamp()
    for e in spec["edges"]:
        if e["kind"] == "transfer":
            ts.setdefault(e["src"], datetime.fromisoformat(e["ts"]).timestamp())

    fwd: dict[str, set] = {i: set() for i in ids}
    rev: dict[str, set] = {i: set() for i in ids}
    for e in spec["edges"]:
        if e["kind"] == "network_link":
            continue
        fwd.setdefault(e["src"], set()).add(e["dst"])
        rev.setdefault(e["dst"], set()).add(e["src"])
    flow = [i for i in ids if by_id[i]["kind"] != "ip"]
    for i in flow:
        if i not in ts:
            near = [ts[t] for t in (fwd[i] | rev[i]) if t in ts]
            ts[i] = min(near) if near else 0.0

    sources = [i for i in flow if not rev.get(i)] or \
        sorted(flow, key=lambda i: (len(rev.get(i, ())), ts[i]))[:1]
    start = min(sources, key=lambda i: ts.get(i, 0.0))

    layer: dict[str, int] = {start: 0}
    frontier = [start]
    while frontier:
        nxt = []
        for u in frontier:
            for v in fwd.get(u, ()):
                if v not in layer:
                    layer[v] = layer[u] + 1
                    nxt.append(v)
        frontier = nxt
    for _ in range(len(flow) + 1):
        for i in flow:
            if i in layer:
                continue
            pl = [layer[p] for p in rev.get(i, ()) if p in layer]
            sl = [layer[s] for s in fwd.get(i, ()) if s in layer]
            if pl:
                layer[i] = max(pl) + 1
            elif sl:
                layer[i] = max(min(sl) - 1, 0)
    for i in flow:
        layer.setdefault(i, 0)

    cols: dict[int, list[str]] = {}
    for i in flow:
        cols.setdefault(layer[i], []).append(i)
    chain = all(len(c) == 1 for c in cols.values()) and len(cols) >= 3
    spec["_chain"] = chain  # render() reads this to place edge labels / IPs

    pos: dict[str, tuple[float, float]] = {}
    col_top: dict[int, float] = {}
    for lx in sorted(cols):
        col = sorted(cols[lx], key=lambda i: (
            {"tx": 0, "more": 1, "wallet": 2}.get(by_id[i]["kind"], 3), ts.get(i, 0.0), i))
        n = len(col)
        for j, i in enumerate(col):
            y = 0.0 if chain else (j - (n - 1) / 2.0) * 1.55
            pos[i] = (float(lx) * (2.6 if chain else 3.2), y)
        col_top[lx] = max(pos[i][1] for i in col)

    for e in spec["edges"]:
        if e["kind"] == "network_link" and e["dst"] in layer:
            lx = layer[e["dst"]]
            pos[e["src"]] = (pos[e["dst"]][0], col_top.get(lx, 0.0) + 2.8)
    return pos


# ---------------------------------------------------------------------------
# 3. render
# ---------------------------------------------------------------------------

_COLOR = {
    "subject": "#e04a4a", "origin": "#f0a33c", "terminal": "#7b61ff",
    "intermediary": "#4a86e8", "ip": "#43b581", "tx": "#b8b8b8", "more": "#d8d8d8",
}


def _fmt(a) -> str:
    return f"{a:.4f}".rstrip("0").rstrip(".") if a is not None else ""


def render(spec: dict, out_path: str, pattern_label: str | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    pos = choose_layout(spec)
    chain = spec.get("_chain", False)
    spine_ids = set(spec.get("spine_ids", ()))

    xs = [p[0] for p in pos.values()] or [0.0]
    ys = [p[1] for p in pos.values()] or [0.0]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    has_return = any(pos.get(e["dst"], (0,))[0] < pos.get(e["src"], (1,))[0] - 0.01
                     for e in spec["edges"] if e["kind"] != "network_link"
                     and e["src"] in pos and e["dst"] in pos)
    span_y += 3.5 if has_return else 0.0
    fig_w = min(max(9.0, span_x * (0.72 if chain else 0.60) + 4), 24)
    fig_h = min(max(4.2 if chain else 6.5, span_y * 0.70 + (3.4 if chain else 3.5)), 15)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    arc_peak = max(ys)
    for e in (e for e in spec["edges"] if e["kind"] != "network_link"):
        if e["src"] not in pos or e["dst"] not in pos:
            continue
        (x1, y1), (x2, y2) = pos[e["src"]], pos[e["dst"]]
        backward = x2 < x1 - 0.01
        # keep the return arc's bulge roughly constant regardless of how far
        # back it reaches (a long chord with a big rad flies off the canvas)
        rad = min(0.55, 3.6 / max(abs(x2 - x1), 1.0)) if backward else 0.05
        if backward:
            arc_peak = max(arc_peak, max(y1, y2) + rad * abs(x2 - x1) * 0.55 + 1.0)
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>",
            mutation_scale=15, shrinkA=15, shrinkB=15, zorder=1,
            lw=2.2 if e.get("spine") else 1.3,
            color="#c0392b" if backward else "#54636f",
            connectionstyle=f"arc3,rad={rad}",
        ))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if e["kind"] == "transfer":
            parts = [e["txid"]]
            if e.get("geo"):
                parts.append(e["geo"])
            if e.get("amount") is not None:
                parts.append(_fmt(e["amount"]))
            ly = (arc_peak * 0.62 if backward else (my + (0.85 if chain else 0.4)))
            ax.text(mx, ly, "  ·  ".join(parts), fontsize=7.5, ha="center",
                    color="#2b4a63", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#f4f8fb", ec="#c7d6e2", alpha=0.97))
            if e.get("ip"):  # relay IP as a small labelled node just below the hop
                ix = x1 + (x2 - x1) * 0.42
                iy = (min(y1, y2) - 1.15) if chain else (my - 0.75)
                ax.scatter([ix], [iy], s=110, marker="^", c=_COLOR["ip"],
                           edgecolors="#2f6b4f", linewidths=0.7, zorder=4)
                ax.text(ix, iy - 0.3, e["ip"], fontsize=6.3, ha="center", va="top",
                        color="#1d5c3f", zorder=6)
        elif e.get("amount") is not None and (e.get("spine") or backward):
            ax.text(mx, my + (0.6 if backward else 0.32), _fmt(e["amount"]),
                    fontsize=7, ha="center", color="#2b4a63", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    for e in spec["edges"]:  # relay-IP dashed links for fan-out transactions
        if e["kind"] != "network_link" or e["src"] not in pos or e["dst"] not in pos:
            continue
        (x1, y1), (x2, y2) = pos[e["src"]], pos[e["dst"]]
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
            shrinkA=12, shrinkB=12, lw=0.9, color="#93a7ba",
            linestyle=(0, (2, 2)), zorder=1))

    for n in spec["nodes"]:
        if n["id"] not in pos:
            continue
        x, y = pos[n["id"]]
        kind, role = n["kind"], n.get("role", n["kind"])
        if kind == "tx":
            ax.scatter([x], [y], s=210, marker="s", c=_COLOR["tx"],
                       edgecolors="#555", linewidths=0.7, zorder=3)
            lbl = f"{n['label']}\n{n['n_in']} in → {n['n_out']} out"
            ax.text(x, y - 0.66, lbl, fontsize=7, ha="center", va="top", color="#333", zorder=4)
        elif kind == "more":
            ax.scatter([x], [y], s=190, marker="o", c=_COLOR["more"],
                       edgecolors="#999", linewidths=0.6, zorder=3)
            t = n["label"] + (f"\n≈ {_fmt(n['each'])} ea" if n.get("each") else "")
            ax.text(x, y - 0.52, t, fontsize=6.8, ha="center", va="top",
                    color="#777", style="italic", zorder=4)
        elif kind == "ip":  # only fan-tx relay IPs reach here (transfer IPs drawn above)
            ax.scatter([x], [y], s=230, marker="^", c=_COLOR["ip"],
                       edgecolors="#2f6b4f", linewidths=0.8, zorder=3)
            ax.text(x, y + 0.5, n["label"], fontsize=7.5, ha="center", va="bottom",
                    color="#1d5c3f", fontweight="bold", zorder=4)
        else:
            on_spine = n["id"] in spine_ids
            big = role in ("subject", "origin", "terminal")
            ax.scatter([x], [y],
                       s=520 if role == "subject" else (330 if big else (220 if on_spine else 150)),
                       c=_COLOR.get(role, _COLOR["intermediary"]),
                       edgecolors="#222", linewidths=1.1 if (big or on_spine) else 0.5, zorder=3)
            fs = 8.5 if role == "subject" else (7.3 if big else (6.8 if on_spine else 6.3))
            ax.text(x, y - (0.55 if role == "subject" else 0.46), n["label"],
                    fontsize=fs, ha="center", va="top",
                    fontweight="bold" if role == "subject" else "normal",
                    color="#111" if (big or on_spine) else "#555", zorder=5)

    title = f"Investigative subgraph — {spec['entity'][:14]}…"
    if pattern_label and pattern_label not in ("none", "unknown"):
        title += f"    ({pattern_label.replace('_', ' ')})"
    ax.set_title(title, fontsize=12, pad=14)
    counts = f"{spec['n_wallets']} wallets · {spec['n_tx']} transactions · {spec['n_ips']} relay IPs"
    if spec.get("collapsed"):
        counts += f"   ({spec['collapsed']} fan-out recipients collapsed)"
    legend = ("red = subject   orange = ring start   purple = terminal   "
              "blue = intermediary   green ▲ = relay IP   grey ■ = fan-out tx   "
              "curved red arrow = funds returning to start")
    ax.text(0.5, -0.03, counts, transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color="#445")
    ax.text(0.5, -0.075, legend, transform=ax.transAxes, ha="center", va="top",
            fontsize=7, color="#889")
    x_pad = (max(xs) - min(xs)) * 0.10 + 1.2
    ip_below = any(e["kind"] == "transfer" and e.get("ip") for e in spec["edges"])
    y_low = min(ys) - (2.4 if ip_below else 1.3)
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(y_low - 0.8, arc_peak + 1.0)
    ax.axis("off")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _demo() -> None:
    import case_file

    for wallets in (["1L4t1PEf2ocQ1SZdP7BXE6PEPd0152"], ["1xNpddVhHmSkSsxLSNmF3veuvS0335"]):
        cf = case_file.generate("data/synthetic_transactions.csv",
                                "data/synthetic_labels.csv", wallets=wallets)[0]
        spec = cf["subgraph"]
        out = f"output/subgraph_{wallets[0][:10]}.png"
        render(spec, out, cf.get("ground_truth_label"))
        print(f"{wallets[0][:14]}  {cf['ground_truth_label']:15s} "
              f"-> {spec['n_wallets']}w / {spec['n_tx']}tx / {spec['n_ips']}ip "
              f"(collapsed {spec.get('collapsed', 0)})  -> {out}")


if __name__ == "__main__":
    _demo()
