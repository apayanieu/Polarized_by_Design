#!/usr/bin/env python3
"""
Plot the three-phase nullcline experiment.

Reads `data/runs.csv` (produced by `run_nullcline_experiment.py`) and writes
figures to `figures/`:

    fig_heatmap_<top>.png     4 panels (k = 4, 6, 8, 10): frac_polarized
                              heatmap + solid white boundary (always polarized,
                              frac >= 1.0) + dashed white boundary (>= 20 %).
    fig_lines_<top>.png       Solid lines only (the four k curves overlaid),
                              gamma-style.
    fig_all_heatmaps.png      3 x 4 grid of every heatmap.
    fig_lines_all.png         3 panels (one per topology), solid lines only.

The boundary lines are extracted on the 25 x 25 coarse grid: for each `a`
column, the solid line sits at the lowest `S` such that every cell with S' >= S
is fully polarized (frac == 1.0); the dashed line at the lowest `S` such that
every cell with S' >= S has frac >= 0.20. The Phase-3 boundary33 sub-cells
refine the line position by averaging into their parent coarse cell.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Re-use the experiment's config so the plot matches the data 1:1.
from run_nullcline_experiment import (                                          # noqa: E402
    TOPOLOGIES, DEGREES, GRID as DEFAULT_GRID, A_RANGE, S_RANGE,
    P_THRESHOLD, MEAN_HIGH, CSV_PATH,
)

# GRID is inferred from the CSV (`--quick` runs use GRID=8). Set in main().
GRID = DEFAULT_GRID

FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FRAC_SOLID = 1.0     # solid line: every cell above must be fully polarized
FRAC_DASHED = 0.20   # dashed line: every cell above must have >= 20 % polarized

TOP_LABEL = {"er": "Erdős–Rényi", "ws": "Watts–Strogatz", "ba": "Barabási–Albert"}
K_COLORS = {4: "#1f77b4", 6: "#2ca02c", 8: "#d62728", 10: "#9467bd"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _make_grids():
    a_vals = np.geomspace(A_RANGE[0], A_RANGE[1], GRID)
    S_vals = np.geomspace(S_RANGE[0], S_RANGE[1], GRID)
    return a_vals, S_vals


def load_runs(path=CSV_PATH):
    """Return list of dicts (one per simulation row), with numeric fields parsed."""
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV not found: {path}. Run run_nullcline_experiment.py first.")
    out = []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            out.append({
                "topology": row["topology"],
                "k": int(row["k"]),
                "run_kind": row["run_kind"],
                "ia": int(row["ia"]),
                "iS": int(row["iS"]),
                "a": float(row["a"]),
                "S": float(row["S"]),
                "P_final": float(row["P_final"]),
            })
    return out


def aggregate_grid(runs, topology, k):
    """Per coarse cell (ia, iS) in [0, GRID)^2:
        frac_polarized = mean(P_final >= P_THRESHOLD) over the best available kind
        mean_P         = mean(P_final)             over the best available kind

    Priority for the coarse-cell value: long > coarse (Phase 3 sub-cells live
    at encoded ia/iS = parent*10 + jitter and are folded back into their
    parent here, augmenting the run count without displacing the grid).

    Returns (frac_grid, mean_grid), each shape (GRID, GRID) with NaN for
    missing cells; iS is the row index, ia the column.
    """
    by_cell_kind = {}  # (ia, iS, kind) -> list[P_final]
    boundary_by_parent = {}  # (parent_ia, parent_iS) -> list[P_final]

    for r in runs:
        if r["topology"] != topology or r["k"] != k:
            continue
        kind = r["run_kind"]
        if kind == "boundary33":
            # Encoded keys: sub_ia = parent * 10 + jdia; we only care about parent.
            parent_ia = r["ia"] // 10
            parent_iS = r["iS"] // 10
            boundary_by_parent.setdefault((parent_ia, parent_iS), []).append(r["P_final"])
        else:
            by_cell_kind.setdefault((r["ia"], r["iS"], kind), []).append(r["P_final"])

    frac = np.full((GRID, GRID), np.nan)
    mean_P = np.full((GRID, GRID), np.nan)

    cells = {(ia, iS) for (ia, iS, _) in by_cell_kind.keys()}
    cells.update(boundary_by_parent.keys())
    for (ia, iS) in cells:
        if not (0 <= ia < GRID and 0 <= iS < GRID):
            continue
        if (ia, iS, "long") in by_cell_kind:
            ps = list(by_cell_kind[(ia, iS, "long")])
        elif (ia, iS, "coarse") in by_cell_kind:
            ps = list(by_cell_kind[(ia, iS, "coarse")])
        else:
            ps = []
        ps.extend(boundary_by_parent.get((ia, iS), []))
        if not ps:
            continue
        arr = np.asarray(ps)
        frac[iS, ia] = float((arr >= P_THRESHOLD).mean())
        mean_P[iS, ia] = float(arr.mean())
    return frac, mean_P


# ---------------------------------------------------------------------------
# Boundary extraction
# ---------------------------------------------------------------------------

def boundary_line(frac, S_vals, thresh):
    """For each column ia, find the lowest S such that every cell with S' >= S
    has frac >= thresh. Returns an array of length GRID (NaN if column never
    qualifies). NaN cells are treated as failing the threshold.
    """
    out = np.full(frac.shape[1], np.nan)
    n_S = frac.shape[0]
    for ia in range(frac.shape[1]):
        col = frac[:, ia]
        # Scan top-down (high iS first). The boundary is the lowest iS such
        # that col[iS:] are all valid and all >= thresh.
        for iS in range(n_S):
            sub = col[iS:]
            if np.all(np.isfinite(sub)) and np.all(sub >= thresh):
                out[ia] = S_vals[iS]
                break
    return out


# ---------------------------------------------------------------------------
# Plot primitives
# ---------------------------------------------------------------------------

def _draw_heatmap(ax, frac, a_vals, S_vals, draw_lines=True, title=None):
    """Plot one heatmap panel. Returns the QuadMesh for colorbar attachment."""
    # pcolormesh wants edges; build log-spaced edges around the cell centers.
    def edges(v):
        lv = np.log(v)
        d = lv[1] - lv[0]
        return np.exp(np.concatenate([[lv[0] - d / 2], lv + d / 2]))

    a_edges = edges(a_vals)
    S_edges = edges(S_vals)
    mesh = ax.pcolormesh(
        a_edges, S_edges, np.ma.masked_invalid(frac),
        cmap="viridis", vmin=0.0, vmax=1.0, shading="auto",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("a")
    ax.set_ylabel("S")
    if title:
        ax.set_title(title)

    has_label = False
    if draw_lines:
        solid = boundary_line(frac, S_vals, FRAC_SOLID)
        dashed = boundary_line(frac, S_vals, FRAC_DASHED)
        m_solid = np.isfinite(solid)
        m_dashed = np.isfinite(dashed)
        if m_solid.any():
            ax.plot(a_vals[m_solid], solid[m_solid], color="white", lw=2.0,
                    label=f"frac ≥ {FRAC_SOLID:.2f}")
            has_label = True
        if m_dashed.any():
            ax.plot(a_vals[m_dashed], dashed[m_dashed], color="white", lw=1.5,
                    ls="--", label=f"frac ≥ {FRAC_DASHED:.2f}")
            has_label = True
    return mesh, has_label


def _draw_lines_panel(ax, runs, topology, a_vals, S_vals, title=None):
    """Solid-line-only panel: one curve per k, all overlaid."""
    drew_any = False
    for k in DEGREES:
        frac, _ = aggregate_grid(runs, topology, k)
        if not np.isfinite(frac).any():
            continue
        line = boundary_line(frac, S_vals, FRAC_SOLID)
        m = np.isfinite(line)
        if not m.any():
            continue
        ax.plot(a_vals[m], line[m], color=K_COLORS.get(k, None),
                lw=2.0, label=f"k = {k}")
        drew_any = True
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(a_vals[0], a_vals[-1])
    ax.set_ylim(S_vals[0], S_vals[-1])
    ax.set_xlabel("a")
    ax.set_ylabel("S")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    if drew_any:
        ax.legend(loc="best", fontsize=8)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_heatmap_one_topology(runs, topology, a_vals, S_vals):
    fig, axes = plt.subplots(1, len(DEGREES), figsize=(4.5 * len(DEGREES), 4.5),
                             constrained_layout=True)
    if len(DEGREES) == 1:
        axes = [axes]
    last_mesh = None
    for ax, k in zip(axes, DEGREES):
        frac, _ = aggregate_grid(runs, topology, k)
        last_mesh, has_label = _draw_heatmap(
            ax, frac, a_vals, S_vals,
            title=f"{TOP_LABEL.get(topology, topology)}  k={k}",
        )
        if has_label:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=axes, shrink=0.85)
        cbar.set_label("frac_polarized")
    out = FIG_DIR / f"fig_heatmap_{topology}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_lines_one_topology(runs, topology, a_vals, S_vals):
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    _draw_lines_panel(ax, runs, topology, a_vals, S_vals,
                      title=f"{TOP_LABEL.get(topology, topology)} — solid boundary")
    out = FIG_DIR / f"fig_lines_{topology}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_all_heatmaps(runs, a_vals, S_vals):
    n_top = len(TOPOLOGIES)
    n_k = len(DEGREES)
    fig, axes = plt.subplots(n_top, n_k, figsize=(4 * n_k, 3.8 * n_top),
                             constrained_layout=True, squeeze=False)
    last_mesh = None
    for i, top in enumerate(TOPOLOGIES):
        for j, k in enumerate(DEGREES):
            frac, _ = aggregate_grid(runs, top, k)
            last_mesh, _ = _draw_heatmap(
                axes[i, j], frac, a_vals, S_vals,
                title=f"{TOP_LABEL.get(top, top)}  k={k}",
            )
    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=axes, shrink=0.85)
        cbar.set_label("frac_polarized")
    out = FIG_DIR / "fig_all_heatmaps.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_lines_all(runs, a_vals, S_vals):
    fig, axes = plt.subplots(1, len(TOPOLOGIES), figsize=(5.5 * len(TOPOLOGIES), 5),
                             constrained_layout=True, squeeze=False)
    for ax, top in zip(axes[0], TOPOLOGIES):
        _draw_lines_panel(ax, runs, top, a_vals, S_vals,
                          title=TOP_LABEL.get(top, top))
    out = FIG_DIR / "fig_lines_all.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    runs = load_runs()
    print(f"[plot] loaded {len(runs)} rows from {CSV_PATH}")

    # Infer grid size from coarse-phase rows (smoke tests use GRID=8, full runs
    # use 25). max(ia) over kind=='coarse' is GRID-1 because Phase 1 visits
    # every cell.
    coarse_ia = [r["ia"] for r in runs if r["run_kind"] == "coarse"]
    if not coarse_ia:
        print("[plot] no coarse-phase rows; cannot infer grid")
        return
    grid_size = max(coarse_ia) + 1
    globals()["GRID"] = grid_size
    a_vals, S_vals = _make_grids()
    print(f"[plot] inferred grid={grid_size}")

    # Restrict TOPOLOGIES / DEGREES to those actually present in the CSV so
    # partial runs don't produce empty panels.
    present_top = {r["topology"] for r in runs}
    present_k = {r["k"] for r in runs}
    tops = [t for t in TOPOLOGIES if t in present_top]
    ks = [k for k in DEGREES if k in present_k]
    if not tops or not ks:
        print("[plot] no data to plot")
        return
    globals()["TOPOLOGIES"] = tops
    globals()["DEGREES"] = ks

    print(f"[plot] topologies={tops}  degrees={ks}")
    for top in tops:
        fig_heatmap_one_topology(runs, top, a_vals, S_vals)
        fig_lines_one_topology(runs, top, a_vals, S_vals)

    if len(tops) > 1 or len(ks) > 1:
        fig_all_heatmaps(runs, a_vals, S_vals)
    if len(tops) > 1:
        fig_lines_all(runs, a_vals, S_vals)

    print(f"[plot] figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
