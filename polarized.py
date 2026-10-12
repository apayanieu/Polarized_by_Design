#!/usr/bin/env python3
"""
Interactive polarisation model: competing persuasion + adaptive social weights.
Uses stubbornness S, polarization coefficient P(t), and configurable initial
fraction of Debunkers (pD).

Run:
    python toy_model4.py

Dependencies:
    pip install numpy networkx matplotlib
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple
import numpy as np

try:
    import networkx as nx
except ImportError:
    raise SystemExit("networkx required. Install with: pip install networkx")

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection
    from matplotlib.widgets import Button, Slider, RadioButtons
except ImportError:
    raise SystemExit("matplotlib required. Install with: pip install matplotlib")


# -------------------------------------------------------------------
# Graph factory (unchanged)
# -------------------------------------------------------------------
def make_graph(kind: str, n: int, k: int, p_ws: float = 0.1, seed: int = 1) -> nx.Graph:
    """Generate a synthetic graph of given type."""
    if n < 2:
        raise ValueError("n must be >= 2")
    if k < 1:
        k = 1

    if kind == "er":
        p = min(max(k / (n - 1), 0.0), 1.0)
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        return G

    elif kind == "ws":
        k_ws = k if k % 2 == 0 else k + 1
        k_ws = min(k_ws, n - 1)
        if n == 2:
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edge(0, 1)
            return G
        if k_ws < 2:
            k_ws = 2
        return nx.watts_strogatz_graph(n, k_ws, p_ws, seed=seed)

    elif kind == "ba":
        m = max(1, k // 2)
        m = min(m, n - 1)
        return nx.barabasi_albert_graph(n, m, seed=seed)

    raise ValueError(f"Unknown graph kind: {kind}")


# -------------------------------------------------------------------
# Dynamics – now with stubbornness S
# -------------------------------------------------------------------
def compute_flip_prob(node: int, G: nx.Graph, state: np.ndarray, S: float) -> float:
    """Probability that node flips its opinion (based on all disagreeing neighbours)."""
    prod = 1.0
    for nbr in G.neighbors(node):
        if state[nbr] != state[node]:           # disagree
            w = G[node][nbr]["weight"]
            prod *= (1.0 - w / S)
            if prod == 0.0:
                break
    return 1.0 - prod


def step_opinions(G: nx.Graph, state: np.ndarray, S: float, rng: np.random.Generator) -> int:
    """Synchronous opinion update. Returns number of flips."""
    n = state.size
    flip_probs = np.array([compute_flip_prob(i, G, state, S) for i in range(n)])
    flips = 0
    for i in range(n):
        if rng.random() < flip_probs[i]:
            state[i] ^= 1          # toggle 0<->1
            flips += 1
    return flips


def step_weights(G: nx.Graph, state: np.ndarray, a: float) -> None:
    """Update edge weights: +a for agreement, -a for disagreement, clipped to [0,1]."""
    for u, v in G.edges():
        w = G[u][v]["weight"]
        delta = a if state[u] == state[v] else -a
        new = w + delta
        if new < 0.0:
            new = 0.0
        elif new > 1.0:
            new = 1.0
        G[u][v]["weight"] = new


def initialise(G: nx.Graph, w0: float, pD: float = 0.5, seed: int = None) -> np.ndarray:
    """Assign random opinions (0=F, 1=D) with given fraction pD of D nodes."""
    rng_state = np.random.default_rng(seed)
    state = (rng_state.random(G.number_of_nodes()) < pD).astype(int)
    for u, v in G.edges():
        G[u][v]["weight"] = w0
    return state


# -------------------------------------------------------------------
# Interactive simulator
# -------------------------------------------------------------------
class PolarisationSimulator:
    def __init__(self, graph_kind="ws", n=120, k=8, w0=0.5, S=5.0, a=0.05, pD=0.5, seed=1):
        self.graph_kind = graph_kind
        self.n = n
        self.k = k
        self.w0 = w0
        self.S = S
        self.a = a
        self.pD = pD
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._build_graph()
        self._setup_figure()
        self._create_widgets()

        self.t = 0
        self.running = True
        self.history = {"t": [], "P": []}

        self.anim = FuncAnimation(self.fig, self._update, interval=60, blit=False)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _build_graph(self, seed_offset=0):
        G = make_graph(self.graph_kind, self.n, self.k, seed=self.seed + seed_offset)
        self.G = G
        self.state = initialise(G, self.w0, pD=self.pD, seed=self.seed + seed_offset)
        self.pos = nx.spring_layout(G, seed=42)
        self.pos_arr = np.array([self.pos[u] for u in G.nodes()])

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(12, 6))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 2])

        self.ax_net = self.fig.add_subplot(gs[0, 0])
        self.ax_net.set_xticks([])
        self.ax_net.set_yticks([])
        self.ax_net.set_title("Opinion network (red=F, blue=D)  edge thickness ∝ weight")

        self.edge_lc = LineCollection([], linewidths=0.5, alpha=0.3)
        self.ax_net.add_collection(self.edge_lc)

        colors = ["tab:red" if s == 0 else "tab:blue" for s in self.state]
        self.node_scat = self.ax_net.scatter(
            self.pos_arr[:, 0], self.pos_arr[:, 1], s=40,
            c=colors, edgecolors="k", linewidths=0.3, zorder=3
        )

        self.txt_info = self.ax_net.text(0.02, 0.98, "", transform=self.ax_net.transAxes,
                                         va="top", ha="left", fontsize=9)

        self.ax_ts = self.fig.add_subplot(gs[0, 1])
        self.ax_ts.set_title("Polarization coefficient P(t)")
        self.ax_ts.set_xlabel("t")
        self.ax_ts.set_ylabel("P(t)")
        self.ax_ts.set_ylim(0, 1)

        self.line_P, = self.ax_ts.plot([], [], label="P(t)", color="black", lw=2)
        self.ax_ts.legend(loc="upper right")

    def _create_widgets(self):
        axcolor = "lightgoldenrodyellow"
        left, width = 0.08, 0.25
        y0, dy, h = 0.07, 0.04, 0.025

        # Sliders – pD sits at the bottom
        ax_pD     = plt.axes([left, y0 - dy,      width, h], facecolor=axcolor)
        ax_S      = plt.axes([left, y0 + dy*4,    width, h], facecolor=axcolor)
        ax_a      = plt.axes([left, y0 + dy*3,    width, h], facecolor=axcolor)
        ax_w0     = plt.axes([left, y0 + dy*2,    width, h], facecolor=axcolor)
        ax_k      = plt.axes([left, y0 + dy*1,    width, h], facecolor=axcolor)
        ax_n      = plt.axes([left, y0 + dy*0,    width, h], facecolor=axcolor)

        self.sl_pD = Slider(ax_pD, "% of D", 0.0, 1.0, valinit=self.pD, valstep=0.05)
        self.sl_S  = Slider(ax_S,  "S", 1.0, 100.0, valinit=self.S, valstep=1.0)
        self.sl_a  = Slider(ax_a,  "a",              0.0,   0.3, valinit=self.a, valstep=0.01)
        self.sl_w0 = Slider(ax_w0, "w0",             0.0,   1.0, valinit=self.w0, valstep=0.05)
        self.sl_k  = Slider(ax_k,  "k (avg deg)",      2,    20, valinit=self.k, valstep=1)
        self.sl_n  = Slider(ax_n,  "N (next)",         30,   400, valinit=self.n, valstep=10)

        # Radio buttons
        rax = plt.axes([0.40, y0 + dy*3, 0.18, 0.12], facecolor=axcolor)
        label_order = ["ws", "er", "ba"]
        active_idx = label_order.index(self.graph_kind) if self.graph_kind in label_order else 0
        self.radio = RadioButtons(rax, tuple(label_order), active=active_idx)
        self.radio.on_clicked(self._on_graph_type)

        # Buttons
        ax_pause = plt.axes([0.62, y0 + dy*3, 0.09, 0.04])
        ax_reset = plt.axes([0.73, y0 + dy*3, 0.09, 0.04])
        ax_new   = plt.axes([0.62, y0 + dy*1.5, 0.20, 0.04])

        self.btn_pause = Button(ax_pause, "Pause")
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_new   = Button(ax_new, "New Graph")

        self.btn_pause.on_clicked(self._toggle_running)
        self.btn_reset.on_clicked(self._reset)
        self.btn_new.on_clicked(self._new_graph)

    def _on_graph_type(self, label):
        self.graph_kind = label

    def _toggle_running(self, event=None):
        self.running = not self.running
        self.btn_pause.label.set_text("Run" if not self.running else "Pause")

    def _reset(self, event=None):
        self.pD = self.sl_pD.val
        self.state = initialise(self.G, self.w0, pD=self.pD, seed=self.seed)
        self.t = 0
        self.history = {"t": [], "P": []}
        self._update_artists()

    def _new_graph(self, event=None):
        self.n = int(self.sl_n.val)
        self.k = int(self.sl_k.val)
        self.pD = self.sl_pD.val
        self._build_graph(seed_offset=hash(self.t))
        self.state = initialise(self.G, self.w0, pD=self.pD, seed=self.seed)
        self.t = 0
        self.history = {"t": [], "P": []}
        self.pos = nx.spring_layout(self.G, seed=42)
        self.pos_arr = np.array([self.pos[u] for u in self.G.nodes()])
        self.node_scat.set_offsets(self.pos_arr)
        self._update_artists()

    def _on_key(self, event):
        if event.key == " ":
            self._toggle_running()
        elif event.key.lower() == "r":
            self._reset()
        elif event.key.lower() == "n":
            self._new_graph()

    def _update_artists(self):
        # edges
        segs = []
        widths = []
        alphas = []
        for u, v in self.G.edges():
            segs.append(np.array([self.pos[u], self.pos[v]]))
            w = self.G[u][v]["weight"]
            widths.append(w * 2.5)
            alphas.append(0.1 + 0.8 * w)
        self.edge_lc.set_segments(segs)
        self.edge_lc.set_linewidths(widths)
        self.edge_lc.set_alpha(alphas)

        colors = ["tab:red" if s == 0 else "tab:blue" for s in self.state]
        self.node_scat.set_facecolors(colors)

        tarr = np.array(self.history["t"])
        if len(tarr) > 0:
            self.line_P.set_data(tarr, self.history["P"])
            self.ax_ts.set_xlim(0, max(50, tarr[-1]))

        nF = np.sum(self.state == 0)
        nD = np.sum(self.state == 1)
        cross_edges = 0
        sum_cross = 0.0
        for u, v in self.G.edges():
            if self.state[u] != self.state[v]:
                cross_edges += 1
                sum_cross += self.G[u][v]["weight"]
        avg_cross = sum_cross / cross_edges if cross_edges > 0 else 0.0
        P_now = 1.0 - avg_cross

        self.txt_info.set_text(
            f"t={self.t}\n"
            f"F={nF} D={nD}\n"
            f"P(t)={P_now:.3f}\n"
            f"S={self.sl_S.val:.1f}  a={self.sl_a.val:.2f}  pD={self.sl_pD.val:.2f}"
        )
        self.fig.canvas.draw_idle()

    def _record_history(self):
        cross_edges = 0
        sum_cross = 0.0
        for u, v in self.G.edges():
            if self.state[u] != self.state[v]:
                cross_edges += 1
                sum_cross += self.G[u][v]["weight"]
        avg_cross = sum_cross / cross_edges if cross_edges > 0 else 0.0
        P = 1.0 - avg_cross
        self.history["t"].append(self.t)
        self.history["P"].append(P)

    def _update(self, frame):
        self.S = self.sl_S.val
        self.a = self.sl_a.val
        self.w0 = self.sl_w0.val

        if self.running:
            step_opinions(self.G, self.state, self.S, self.rng)
            step_weights(self.G, self.state, self.a)
            self.t += 1

        self._record_history()
        self._update_artists()
        return [self.node_scat, self.edge_lc, self.line_P]

    def show(self):
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=["ws","er","ba"], default="ws")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--S", type=float, default=5.0, help="stubbornness parameter")
    parser.add_argument("--a", type=float, default=0.05)
    parser.add_argument("--pD", type=float, default=0.5, help="initial fraction of D nodes")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    sim = PolarisationSimulator(
        graph_kind=args.graph,
        n=args.n,
        k=args.k,
        S=args.S,
        a=args.a,
        pD=args.pD,
        seed=args.seed
    )
    sim.show()