#!/usr/bin/env python3
"""
Three-phase nullcline experiment for the polarized.py model.

Phase 1 — coarse sweep at T = T_COARSE on a log-spaced 25x25 grid of (a, S)
          for {ER, WS, BA} x {k = 4, 6, 8, 10}.
Phase 2 — re-run cells with mean_P in [MEAN_LOW, MEAN_HIGH) at T = T_LONG.
Phase 3 — for each boundary cell (8-neighborhood disagreement on regime),
          run a 3x3 sub-grid of jittered (a, S) at T = T_LONG.

All individual runs (one row per simulation) are written to a single CSV.
The pipeline is RESUMABLE: re-running this script skips cells already present
in the CSV (matched on topology, k, run_kind, ia, iS, run_id).

Run overnight:
    nohup python run_nullcline_experiment.py > run.log 2>&1 &
    tail -f run.log

CLI:
    --jobs N      number of worker processes (default: cpu_count - 2, capped 8)
    --quick       tiny grid for smoke test
    --topology    restrict to one topology (default: all three)
    --k           restrict to one degree (default: all four)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Local imports (assumes polarized.py and polarized_fast.py are in same dir)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from polarized import make_graph                                              # noqa: E402
from polarized_fast import (                                                  # noqa: E402
    nx_to_arrays,
    step_opinions_fast,
    step_weights_fast,
    polarization_coefficient_fast,
)

# =============================================================================
# Config (edit here, not via CLI, to keep the experiment reproducible)
# =============================================================================
TOPOLOGIES = ["er", "ws", "ba"]
DEGREES = [4, 6, 8, 10]

N_NODES = 200
N_RUNS = 20             # Phases 1 and 2
N_RUNS_BOUNDARY = 35    # Phase 3 (3x3 sub-cells around boundary cells)
GRID = 25
A_RANGE = (1e-3, 0.5)   # log-spaced
S_RANGE = (1.0, 100.0)  # log-spaced

T_COARSE = 1000
T_LONG = 3000

P_THRESHOLD = 0.95      # a single run is "polarized" if P_final >= P_THRESHOLD
MEAN_LOW = 0.80         # cells with mean in [MEAN_LOW, MEAN_HIGH) -> extend to T_LONG
MEAN_HIGH = 0.99

P_D = 0.5
W0 = 0.5

GRAPH_SEED = 42
STATE_SEED = 123
BASE_SEED = 2024

DATA_DIR = HERE / "data"
CSV_PATH = DATA_DIR / "runs.csv"

CSV_FIELDS = [
    "topology", "k", "run_kind", "ia", "iS",
    "a", "S", "T", "run_id", "seed", "P_final", "polarized",
]

# =============================================================================
# Core: single-cell worker
# =============================================================================

def _make_grids(grid=GRID):
    a_vals = np.geomspace(A_RANGE[0], A_RANGE[1], grid)
    S_vals = np.geomspace(S_RANGE[0], S_RANGE[1], grid)
    return a_vals, S_vals


def _run_one_simulation(A_mat, a_val, S_val, T, w0, pD,
                        sim_seed, state_seed) -> float:
    """One realization. Returns final P."""
    N = A_mat.shape[0]
    rng = np.random.default_rng(sim_seed)
    state = (np.random.default_rng(state_seed).random(N) < pD).astype(np.int8)
    W_mat = np.where(A_mat, w0, 0.0).astype(np.float64)
    for _ in range(T):
        state = step_opinions_fast(A_mat, W_mat, state, S_val, rng)
        W_mat = step_weights_fast(A_mat, W_mat, state, a_val)
    return polarization_coefficient_fast(A_mat, W_mat, state)


def run_cell(args):
    """Worker: run N_RUNS independent realizations for a single (topology, k, a, S).

    args is a dict so it serializes cleanly to processes.
    Returns list[dict] of CSV rows.
    """
    topology  = args["topology"]
    k         = args["k"]
    ia        = args["ia"]
    iS        = args["iS"]
    a_val     = args["a_val"]
    S_val     = args["S_val"]
    T         = args["T"]
    run_kind  = args["run_kind"]
    n_runs    = args["n_runs"]
    seed_tag  = args["seed_tag"]    # extra entropy to separate run_kind seed streams

    G = make_graph(topology, N_NODES, k, seed=GRAPH_SEED)
    A_mat, _ = nx_to_arrays(G, w0=W0)

    rows = []
    for r in range(n_runs):
        # Distinct, reproducible seed per (cell, run, run_kind)
        sim_seed   = BASE_SEED + 1_000_003 * ia + 1_009 * iS + 31 * r + seed_tag
        state_seed = STATE_SEED + r * 7 + seed_tag

        P_final = _run_one_simulation(
            A_mat, a_val, S_val, T, W0, P_D, sim_seed, state_seed
        )

        rows.append({
            "topology": topology,
            "k": k,
            "run_kind": run_kind,
            "ia": ia,
            "iS": iS,
            "a": float(a_val),
            "S": float(S_val),
            "T": T,
            "run_id": r,
            "seed": sim_seed,
            "P_final": float(P_final),
            "polarized": int(P_final >= P_THRESHOLD),
        })
    return rows


# =============================================================================
# CSV I/O
# =============================================================================

def init_csv():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()


def append_rows(rows):
    if not rows:
        return
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        for row in rows:
            w.writerow(row)


def load_done_keys():
    """Return set of (topology, k, run_kind, ia, iS, run_id) already in CSV."""
    done = set()
    if not CSV_PATH.exists():
        return done
    with open(CSV_PATH, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            done.add((
                row["topology"], int(row["k"]), row["run_kind"],
                int(row["ia"]), int(row["iS"]), int(row["run_id"]),
            ))
    return done


def load_cell_stats(run_kind_priority=("boundary33", "long", "coarse")):
    """Aggregate per-cell stats from CSV. Returns dict keyed by (topology, k, ia, iS).

    Higher-priority run_kind values override lower ones for the SAME (ia, iS).
    """
    if not CSV_PATH.exists():
        return {}
    runs = {}   # (top, k, ia, iS, run_kind) -> list of P_final
    with open(CSV_PATH, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["topology"], int(row["k"]), int(row["ia"]),
                   int(row["iS"]), row["run_kind"])
            runs.setdefault(key, []).append(float(row["P_final"]))
    # Pick highest-priority run_kind per cell
    stats = {}
    cells = {(t, k, ia, iS) for (t, k, ia, iS, _) in runs.keys()}
    for (t, k, ia, iS) in cells:
        chosen_kind = None
        for kind in run_kind_priority:
            if (t, k, ia, iS, kind) in runs:
                chosen_kind = kind
                break
        if chosen_kind is None:
            continue
        ps = np.asarray(runs[(t, k, ia, iS, chosen_kind)])
        stats[(t, k, ia, iS)] = {
            "mean_P": float(ps.mean()),
            "frac_polarized": float((ps >= P_THRESHOLD).mean()),
            "n_runs": int(ps.size),
            "kind": chosen_kind,
        }
    return stats


# =============================================================================
# Phase planners
# =============================================================================

def plan_phase1(grid=GRID, n_runs=N_RUNS):
    a_vals, S_vals = _make_grids(grid)
    plan = []
    for topology in TOPOLOGIES:
        for k in DEGREES:
            for ia, av in enumerate(a_vals):
                for iS, Sv in enumerate(S_vals):
                    plan.append({
                        "topology": topology, "k": k,
                        "ia": ia, "iS": iS,
                        "a_val": float(av), "S_val": float(Sv),
                        "T": T_COARSE, "run_kind": "coarse",
                        "n_runs": n_runs, "seed_tag": 0,
                    })
    return plan, a_vals, S_vals


def plan_phase2(stats, grid=GRID, n_runs=N_RUNS):
    """Cells with mean_P in [MEAN_LOW, MEAN_HIGH) under run_kind=coarse → re-run long."""
    a_vals, S_vals = _make_grids(grid)
    plan = []
    for (top, k, ia, iS), s in stats.items():
        if s["kind"] != "coarse":
            continue
        if MEAN_LOW <= s["mean_P"] < MEAN_HIGH:
            plan.append({
                "topology": top, "k": k,
                "ia": ia, "iS": iS,
                "a_val": float(a_vals[ia]), "S_val": float(S_vals[iS]),
                "T": T_LONG, "run_kind": "long",
                "n_runs": n_runs, "seed_tag": 11,
            })
    return plan


def _is_always(s):
    return s["frac_polarized"] >= 1.0 and s["mean_P"] >= MEAN_HIGH


def plan_phase3(stats, grid=GRID, n_runs=N_RUNS_BOUNDARY, jitter_frac=0.25):
    """3x3 sub-grid around boundary cells (8-neighborhood regime mismatch).

    Each sub-cell runs `n_runs` simulations (default N_RUNS_BOUNDARY = 35),
    higher than Phases 1 and 2 to tighten the dashed-line ≥20% estimate that
    has Wilson CI [6%, 44%] at 20 runs.

    Jitter is in log space: each sub-cell sits at log(a) +/- delta_a, log(S) +/- delta_S
    where delta_X = jitter_frac * log-spacing.
    """
    a_vals, S_vals = _make_grids(grid)
    log_a = np.log(a_vals)
    log_S = np.log(S_vals)
    da = (log_a[1] - log_a[0]) * jitter_frac
    dS = (log_S[1] - log_S[0]) * jitter_frac

    plan = []
    # Group by (topology, k) so we look at one regime grid at a time
    cells_by_grid = {}
    for (top, k, ia, iS), s in stats.items():
        if s["kind"] == "boundary33":
            continue
        cells_by_grid.setdefault((top, k), {})[(ia, iS)] = s

    for (top, k), cmap in cells_by_grid.items():
        # Find boundary cells: regime differs from at least one neighbor
        for (ia, iS), s in cmap.items():
            my_always = _is_always(s)
            on_boundary = False
            for dia in (-1, 0, 1):
                for diS in (-1, 0, 1):
                    if dia == 0 and diS == 0:
                        continue
                    nb = cmap.get((ia + dia, iS + diS))
                    if nb is None:
                        continue
                    if _is_always(nb) != my_always:
                        on_boundary = True
                        break
                if on_boundary:
                    break
            if not on_boundary:
                continue
            # 3x3 sub-grid
            for jdia, jx in enumerate([-1, 0, 1]):
                for jdiS, jy in enumerate([-1, 0, 1]):
                    sub_a = float(np.exp(log_a[ia] + jx * da))
                    sub_S = float(np.exp(log_S[iS] + jy * dS))
                    # Encode sub-position into (ia, iS) so CSV stays uniquely keyed
                    sub_ia = ia * 10 + jdia      # 0..GRID*10
                    sub_iS = iS * 10 + jdiS
                    plan.append({
                        "topology": top, "k": k,
                        "ia": sub_ia, "iS": sub_iS,
                        "a_val": sub_a, "S_val": sub_S,
                        "T": T_LONG, "run_kind": "boundary33",
                        "n_runs": n_runs, "seed_tag": 23,
                    })
    return plan


def filter_done(plan, done_keys):
    """Drop jobs whose all n_runs runs are already in CSV."""
    out = []
    for job in plan:
        if all(
            (job["topology"], job["k"], job["run_kind"], job["ia"],
             job["iS"], r) in done_keys
            for r in range(job["n_runs"])
        ):
            continue
        out.append(job)
    return out


# =============================================================================
# Driver
# =============================================================================

def execute(plan, n_jobs, label):
    if not plan:
        print(f"[{label}] nothing to do")
        return
    print(f"[{label}] {len(plan)} cells, {n_jobs} workers")
    t0 = time.time()
    done_count = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = [ex.submit(run_cell, job) for job in plan]
        for fut in as_completed(futs):
            rows = fut.result()
            append_rows(rows)
            done_count += 1
            if done_count % max(1, len(plan) // 50) == 0 or done_count == len(plan):
                elapsed = time.time() - t0
                eta = elapsed / done_count * (len(plan) - done_count)
                print(f"[{label}]  {done_count}/{len(plan)}  "
                      f"elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s", flush=True)
    print(f"[{label}] DONE in {time.time() - t0:.1f}s")


def main():
    global TOPOLOGIES, DEGREES, GRID, N_RUNS, N_RUNS_BOUNDARY, T_COARSE, T_LONG

    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--quick", action="store_true",
                   help="Tiny grid for smoke testing.")
    p.add_argument("--topology", choices=TOPOLOGIES, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--skip-phase", action="append", default=[],
                   help="Skip phase: 1, 2, or 3. Can be repeated.")

    args = p.parse_args()

    if args.topology:
        TOPOLOGIES = [args.topology]
    if args.k:
        DEGREES = [args.k]
    if args.quick:
        GRID = 8
        N_RUNS = 5
        N_RUNS_BOUNDARY = 9
        T_COARSE = 200
        T_LONG = 500
        print("[quick] reduced grid for smoke test")

    cores = os.cpu_count() or 2
    if args.jobs is None:
        n_jobs = max(1, min(8, cores - 2))
    else:
        n_jobs = args.jobs

    print(f"[setup] CSV at {CSV_PATH}")
    print(f"[setup] topologies={TOPOLOGIES} degrees={DEGREES} "
          f"grid={GRID} N={N_NODES} runs={N_RUNS} "
          f"T_coarse={T_COARSE} T_long={T_LONG} workers={n_jobs}")

    init_csv()

    # --- Phase 1 ---
    if "1" not in args.skip_phase:
        done = load_done_keys()
        plan1, _, _ = plan_phase1(GRID, N_RUNS)
        plan1 = filter_done(plan1, done)
        execute(plan1, n_jobs, "phase1")

    # --- Phase 2 ---
    if "2" not in args.skip_phase:
        stats = load_cell_stats()
        done = load_done_keys()
        plan2 = plan_phase2(stats, GRID, N_RUNS)
        plan2 = filter_done(plan2, done)
        execute(plan2, n_jobs, "phase2")

    # --- Phase 3 ---
    if "3" not in args.skip_phase:
        stats = load_cell_stats()
        done = load_done_keys()
        plan3 = plan_phase3(stats, GRID, N_RUNS_BOUNDARY, jitter_frac=0.25)
        plan3 = filter_done(plan3, done)
        execute(plan3, n_jobs, "phase3")

    print("[done] all phases finished")


if __name__ == "__main__":
    main()
