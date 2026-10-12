#!/usr/bin/env python3
"""
polarizationviz.py – interactive force-layout visualisation of the polarisation model.

Imports model functions (make_graph, initialise, step_opinions, step_weights)
from polarized.py and adds a dynamic spring-electric layout that makes
emerging echo chambers visually apparent.

Requires: numpy, networkx, matplotlib
"""

from __future__ import annotations

import argparse
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

from polarized import make_graph, initialise, step_opinions, step_weights


# ----------------------------------------------------------------------
# Force-directed layout that reacts to edge weights
# ----------------------------------------------------------------------
class ForceLayout:
    """
    Simple damped spring-electric layout.

    - Springs: each edge (u,v) with weight w wants a rest length L0 = 1 - w,
      so strongly coupled nodes sit close together.
    - Repulsion: all node pairs repel via inverse-square law.
    - Boundary: hard reflective walls keep nodes inside [0, 1]^2.
    """

    def __init__(
        self,
        G: nx.Graph,
        pos: np.ndarray,
        k_spring: float = 3.0,
        k_rep: float = 6.0,
        damping: float = 0.92,
        dt: float = 0.06,
        min_dist: float = 0.04,
        max_force: float = 5.0,
    ):
        self.G = G
        self.pos = pos.copy()
        self.vel = np.zeros_like(pos)
        self.n = pos.shape[0]
        self.k_spring = k_spring
        self.k_rep = k_rep
        self.damping = damping
        self.dt = dt
        self.min_dist = min_dist
        self.max_force = max_force

    def step(self, iterations: int = 1) -> None:
        """Advance the layout by `iterations` steps."""
        for _ in range(iterations):
            forces = np.zeros_like(self.pos)

            # Repulsion between all unordered pairs
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    diff = self.pos[i] - self.pos[j]
                    dist = np.linalg.norm(diff)
                    if dist < self.min_dist:
                        dist = self.min_dist

                    # Softer repulsion than 1/dist^3
                    f = self.k_rep * diff / (dist ** 2)

                    # Cap the force so nodes do not get blasted into corners
                    fnorm = np.linalg.norm(f)
                    if fnorm > self.max_force:
                        f *= self.max_force / fnorm

                    forces[i] += f
                    forces[j] -= f

            # Spring attraction along edges, strength proportional to weight
            for u, v in self.G.edges():
                w = self.G[u][v]["weight"]
                diff = self.pos[v] - self.pos[u]
                dist = np.linalg.norm(diff)
                if dist < self.min_dist:
                    dist = self.min_dist

                rest_len = 1.0 - w
                displacement = dist - rest_len
                force_mag = self.k_spring * displacement
                direction = diff / dist
                f = force_mag * direction

                forces[u] += f
                forces[v] -= f

            # Semi-implicit Euler update
            self.vel = self.damping * (self.vel + self.dt * forces)
            self.pos += self.dt * self.vel

            # Hard reflective boundary conditions on [0,1]^2
            for d in range(2):
                low = self.pos[:, d] < 0.0
                high = self.pos[:, d] > 1.0

                self.pos[low, d] = 0.0
                self.pos[high, d] = 1.0

                self.vel[low, d] *= -0.5
                self.vel[high, d] *= -0.5


# ----------------------------------------------------------------------
# Interactive visualisation with dynamic layout
# ----------------------------------------------------------------------
class PolarisationViz:
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

        self._record_history()
        self._update_artists()

        self.anim = FuncAnimation(self.fig, self._update, interval=60, blit=False)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _build_graph(self, seed_offset=0):
        G = make_graph(self.graph_kind, self.n, self.k, seed=self.seed + seed_offset)
        self.G = G
        self.state = initialise(G, self.w0, pD=self.pD, seed=self.seed + seed_offset)

        init_pos = nx.spring_layout(G, seed=42)
        self.pos_arr = np.array([init_pos[u] for u in G.nodes()], dtype=float)

        # Rescale initial layout to [0, 1]^2
        mins = self.pos_arr.min(axis=0)
        maxs = self.pos_arr.max(axis=0)
        span = np.where(maxs - mins < 1e-12, 1.0, maxs - mins)
        self.pos_arr = (self.pos_arr - mins) / span

        self.force = ForceLayout(G, self.pos_arr)

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(12, 6))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 2])

        self.ax_net = self.fig.add_subplot(gs[0, 0])
        self.ax_net.set_xticks([])
        self.ax_net.set_yticks([])
        self.ax_net.set_title("Opinion network (red=F, blue=D)  edge thickness ∝ weight")
        self.ax_net.set_aspect("equal", adjustable="box")

        self.edge_lc = LineCollection([], linewidths=0.5, alpha=0.3)
        self.ax_net.add_collection(self.edge_lc)

        init_colors = ["tab:red" if s == 0 else "tab:blue" for s in self.state]
        self.node_scat = self.ax_net.scatter(
            self.pos_arr[:, 0],
            self.pos_arr[:, 1],
            s=40,
            c=init_colors,
            edgecolors="k",
            linewidths=0.3,
            zorder=3,
        )

        self.txt_info = self.ax_net.text(
            0.02,
            0.98,
            "",
            transform=self.ax_net.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )

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

        ax_pD = plt.axes([left, y0 - dy, width, h], facecolor=axcolor)
        ax_S = plt.axes([left, y0 + dy * 4, width, h], facecolor=axcolor)
        ax_a = plt.axes([left, y0 + dy * 3, width, h], facecolor=axcolor)
        ax_w0 = plt.axes([left, y0 + dy * 2, width, h], facecolor=axcolor)
        ax_k = plt.axes([left, y0 + dy * 1, width, h], facecolor=axcolor)
        ax_n = plt.axes([left, y0 + dy * 0, width, h], facecolor=axcolor)

        self.sl_pD = Slider(ax_pD, "% of D", 0.0, 1.0, valinit=self.pD, valstep=0.05)
        self.sl_S = Slider(ax_S, "S", 1.0, 100.0, valinit=self.S, valstep=1.0)
        self.sl_a = Slider(ax_a, "a", 0.0, 0.3, valinit=self.a, valstep=0.01)
        self.sl_w0 = Slider(ax_w0, "w0", 0.0, 1.0, valinit=self.w0, valstep=0.05)
        self.sl_k = Slider(ax_k, "k (avg deg)", 2, 20, valinit=self.k, valstep=1)
        self.sl_n = Slider(ax_n, "N (next)", 30, 400, valinit=self.n, valstep=10)

        rax = plt.axes([0.40, y0 + dy * 3, 0.18, 0.12], facecolor=axcolor)
        label_order = ["ws", "er", "ba"]
        active_idx = label_order.index(self.graph_kind) if self.graph_kind in label_order else 0
        self.radio = RadioButtons(rax, tuple(label_order), active=active_idx)
        self.radio.on_clicked(self._on_graph_type)

        ax_pause = plt.axes([0.62, y0 + dy * 3, 0.09, 0.04])
        ax_reset = plt.axes([0.73, y0 + dy * 3, 0.09, 0.04])
        ax_new = plt.axes([0.62, y0 + dy * 1.5, 0.20, 0.04])

        self.btn_pause = Button(ax_pause, "Pause")
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_new = Button(ax_new, "New Graph")

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
        self.w0 = self.sl_w0.val
        self.state = initialise(self.G, self.w0, pD=self.pD, seed=self.seed)
        self.t = 0
        self.history = {"t": [], "P": []}
        self.force.vel = np.zeros_like(self.force.vel)
        self.force.pos = self.pos_arr.copy()
        self._record_history()
        self._update_artists()

    def _new_graph(self, event=None):
        self.n = int(self.sl_n.val)
        self.k = int(self.sl_k.val)
        self.pD = self.sl_pD.val
        self.w0 = self.sl_w0.val

        self._build_graph(seed_offset=self.t + 1)
        self.state = initialise(self.G, self.w0, pD=self.pD, seed=self.seed)
        self.t = 0
        self.history = {"t": [], "P": []}
        self._record_history()
        self._update_artists()

    def _on_key(self, event):
        if event.key == " ":
            self._toggle_running()
        elif event.key.lower() == "r":
            self._reset()
        elif event.key.lower() == "n":
            self._new_graph()

    def _current_polarisation(self) -> float:
        cross_edges = 0
        sum_cross = 0.0
        for u, v in self.G.edges():
            if self.state[u] != self.state[v]:
                cross_edges += 1
                sum_cross += self.G[u][v]["weight"]
        avg_cross = sum_cross / cross_edges if cross_edges > 0 else 0.0
        return 1.0 - avg_cross

    def _record_history(self):
        self.history["t"].append(self.t)
        self.history["P"].append(self._current_polarisation())

    def _update_artists(self):
        segs = []
        widths = []
        colors = []

        for u, v in self.G.edges():
            segs.append(np.array([self.force.pos[u], self.force.pos[v]]))
            w = self.G[u][v]["weight"]
            widths.append(0.5 + 3.0 * w)
            colors.append(plt.cm.Blues(0.2 + 0.8 * w))

        self.edge_lc.set_segments(segs)
        self.edge_lc.set_linewidths(widths)
        self.edge_lc.set_colors(colors)

        self.node_scat.set_offsets(self.force.pos)
        node_colors = ["tab:red" if s == 0 else "tab:blue" for s in self.state]
        self.node_scat.set_facecolors(node_colors)

        tarr = np.array(self.history["t"])
        Parr = np.array(self.history["P"])
        if len(tarr) > 0:
            self.line_P.set_data(tarr, Parr)
            self.ax_ts.set_xlim(0, max(50, int(tarr[-1]) + 5))

        nF = np.sum(self.state == 0)
        nD = np.sum(self.state == 1)
        self.txt_info.set_text(
            f"t={self.t}\n"
            f"F={nF} D={nD}\n"
            f"P(t)={self._current_polarisation():.3f}\n"
            f"S={self.sl_S.val:.1f}  a={self.sl_a.val:.2f}  pD={self.sl_pD.val:.2f}"
        )

        self.ax_net.set_xlim(0, 1)
        self.ax_net.set_ylim(0, 1)
        self.fig.canvas.draw_idle()

    def _update(self, frame):
        self.S = self.sl_S.val
        self.a = self.sl_a.val
        self.w0 = self.sl_w0.val

        if self.running:
            step_opinions(self.G, self.state, self.S, self.rng)
            step_weights(self.G, self.state, self.a)
            self.t += 1

            self.force.step(iterations=12)
            self._record_history()

        self._update_artists()
        return [self.node_scat, self.edge_lc, self.line_P]

    def show(self):
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=["ws", "er", "ba"], default="ws")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--S", type=float, default=5.0, help="stubbornness")
    parser.add_argument("--a", type=float, default=0.05)
    parser.add_argument("--pD", type=float, default=0.5, help="initial fraction of D")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    viz = PolarisationViz(
        graph_kind=args.graph,
        n=args.n,
        k=args.k,
        S=args.S,
        a=args.a,
        pD=args.pD,
        seed=args.seed,
    )
    viz.show()