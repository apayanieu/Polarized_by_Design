#!/usr/bin/env python3
"""
Vectorized core of the polarization model.

Mathematically identical to polarized.py — only the implementation changes:
networkx dict-of-dicts edge lookups are replaced by dense N×N numpy arrays.

State:
    state : (N,) int8        — 0 = F, 1 = D
    A     : (N, N) bool      — adjacency mask (symmetric, diag=False)
    W     : (N, N) float64   — edge weights (symmetric, 0 outside adjacency)

Equivalence with polarized.py is verified in `test_equivalence`.
Run this file directly to execute the tests + a microbenchmark.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from polarized import (
    make_graph,
    step_opinions as step_opinions_nx,
    step_weights as step_weights_nx,
    initialise as initialise_nx,
)


# ---------------------------------------------------------------------------
# Conversion + initialisation
# ---------------------------------------------------------------------------

def nx_to_arrays(G, w0=None):
    """Convert a networkx graph to dense (A, W) numpy arrays.

    If `w0` is None, copy current edge weights from G; otherwise set all edges to w0.
    """
    n = G.number_of_nodes()
    A = np.zeros((n, n), dtype=bool)
    W = np.zeros((n, n), dtype=np.float64)
    for u, v in G.edges():
        A[u, v] = True
        A[v, u] = True
        w = G[u][v]["weight"] if w0 is None else w0
        W[u, v] = w
        W[v, u] = w
    return A, W


def initialise_fast(G, w0, pD, seed):
    """Drop-in for polarized.initialise — same opinion draw, same weights, but
    returns (state, A, W) numpy arrays instead of mutating G.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    state = (rng.random(n) < pD).astype(np.int8)
    A, W = nx_to_arrays(G, w0=w0)
    return state, A, W


# ---------------------------------------------------------------------------
# Synchronous dynamics — exactly equivalent to polarized.step_*
# ---------------------------------------------------------------------------

def step_opinions_fast(A, W, state, S, rng):
    """Synchronous flip update. Returns new state (int8)."""
    # Disagreement matrix D[i,j] = (state[i] != state[j])
    D = state[:, None] != state[None, :]
    # M[i,j] = edge exists AND nodes disagree
    M = A & D
    # factor[i,j] = (1 - W[i,j]/S) where M else 1.0  (1 → no contribution to product)
    factor = np.where(M, 1.0 - W / S, 1.0)
    # p_flip[i] = 1 - prod_j factor[i,j]
    p_flip = 1.0 - factor.prod(axis=1)
    # Synchronous flips: draw N random numbers in the SAME order as the original
    flips = rng.random(state.size) < p_flip
    return np.where(flips, 1 - state, state).astype(np.int8)


def step_weights_fast(A, W, state, a):
    """In-place edge weight update (+a agreement, -a disagreement, clip [0,1]).

    Returns the (mutated) W for convenience.
    """
    D = state[:, None] != state[None, :]
    # +a if agree, -a if disagree — but only on edges (A True)
    delta = np.where(D, -a, a) * A
    W += delta
    np.clip(W, 0.0, 1.0, out=W)
    W *= A  # ensure non-edge entries stay exactly 0
    return W


def polarization_coefficient_fast(A, W, state):
    """P(t) = 1 - mean weight on cross-opinion edges.

    Convention (matches polarized.py / heatmap_polarized.py): when no
    cross-opinion edges exist, return 1.0 (fully polarised).
    """
    D = state[:, None] != state[None, :]
    cross = A & D
    # Upper triangle to count each edge once
    upper = np.triu(cross, k=1)
    if not upper.any():
        return 1.0
    return 1.0 - float(W[upper].mean())


# ---------------------------------------------------------------------------
# Equivalence test against polarized.py
# ---------------------------------------------------------------------------

def _polarization_coefficient_nx(G, state):
    cross_w = [G[u][v]["weight"] for u, v in G.edges() if state[u] != state[v]]
    return 1.0 - float(np.mean(cross_w)) if cross_w else 1.0


def test_equivalence(n=50, k=8, T=50, S=5.0, a=0.1,
                     init_seed=100, sim_seed=200, graph_seed=42,
                     p_tol=1e-10, verbose=True):
    """Strict step-by-step equivalence test.

    For T steps, both implementations should give identical states (bit-exact)
    and P(t) values matching within `p_tol`. Float ordering can produce
    sub-ULP differences in P(t); for state we expect exact match unless a
    rng-draw lands inside that tiny window.
    """
    # Same graph (built independently — must be identical because make_graph is deterministic)
    G_nx = make_graph("er", n, k, seed=graph_seed)
    G_fast = make_graph("er", n, k, seed=graph_seed)

    # Identical initial state (both use polarized.initialise)
    state_nx = initialise_nx(G_nx, w0=0.5, pD=0.5, seed=init_seed)
    state_fast = state_nx.copy().astype(np.int8)

    # Identical initial weights — fast version reads them from G after initialise_nx ran
    # (but G_fast hasn't run initialise yet; initialise also resets weights on G).
    initialise_nx(G_fast, w0=0.5, pD=0.5, seed=init_seed)  # to set weights on G_fast
    A_fast, W_fast = nx_to_arrays(G_fast)

    # Two RNGs with same seed (they will consume random numbers in the same order)
    rng_nx = np.random.default_rng(sim_seed)
    rng_fast = np.random.default_rng(sim_seed)

    P_nx_arr = np.empty(T)
    P_fast_arr = np.empty(T)
    state_mismatch_step = -1

    for t in range(T):
        step_opinions_nx(G_nx, state_nx, S, rng_nx)
        step_weights_nx(G_nx, state_nx, a)
        P_nx_arr[t] = _polarization_coefficient_nx(G_nx, state_nx)

        state_fast = step_opinions_fast(A_fast, W_fast, state_fast, S, rng_fast)
        W_fast = step_weights_fast(A_fast, W_fast, state_fast, a)
        P_fast_arr[t] = polarization_coefficient_fast(A_fast, W_fast, state_fast)

        if state_mismatch_step < 0 and not np.array_equal(state_nx, state_fast):
            state_mismatch_step = t

    p_diffs = np.abs(P_nx_arr - P_fast_arr)
    max_p_diff = float(p_diffs.max())
    # Compare final weights too
    W_nx_dense = nx_to_arrays(G_nx)[1]
    w_max_diff = float(np.abs(W_nx_dense - W_fast).max())

    passed = (state_mismatch_step < 0) and (max_p_diff < p_tol) and (w_max_diff < 1e-12)

    if verbose:
        print(f"[test] N={n}, k={k}, T={T}, S={S}, a={a}, graph_seed={graph_seed}")
        print(f"       max |P_fast - P_nx|     = {max_p_diff:.2e}  (tol {p_tol:.0e})")
        print(f"       max |W_fast - W_nx|     = {w_max_diff:.2e}")
        print(f"       state mismatch first at = {state_mismatch_step}  (-1 means never)")
        print(f"       result: {'PASS' if passed else 'FAIL'}")
    return passed, max_p_diff, w_max_diff, state_mismatch_step


# ---------------------------------------------------------------------------
# Microbenchmark — fast vs networkx for a single cell
# ---------------------------------------------------------------------------

def _bench_nx(n, T, runs, S=5.0, a=0.1):
    G = make_graph("er", n, 8, seed=42)
    P_final = np.empty(runs)
    for r in range(runs):
        rng = np.random.default_rng(2024 + r)
        state = initialise_nx(G, 0.5, pD=0.5, seed=123 + r)
        traj = np.empty(T)
        for t in range(T):
            step_opinions_nx(G, state, S, rng)
            step_weights_nx(G, state, a)
            traj[t] = _polarization_coefficient_nx(G, state)
        P_final[r] = traj[-1]
    return P_final.mean()


def _bench_fast(n, T, runs, S=5.0, a=0.1):
    G = make_graph("er", n, 8, seed=42)
    P_final = np.empty(runs)
    for r in range(runs):
        rng = np.random.default_rng(2024 + r)
        state, A, W = initialise_fast(G, 0.5, pD=0.5, seed=123 + r)
        traj = np.empty(T)
        for t in range(T):
            state = step_opinions_fast(A, W, state, S, rng)
            W = step_weights_fast(A, W, state, a)
            traj[t] = polarization_coefficient_fast(A, W, state)
        P_final[r] = traj[-1]
    return P_final.mean()


def benchmark():
    print()
    print("=" * 60)
    print("Microbenchmark — networkx vs vectorized")
    print("=" * 60)
    for n in [100, 200, 300]:
        t0 = time.time()
        P_nx = _bench_nx(n, T=1000, runs=20)
        t_nx = time.time() - t0
        t0 = time.time()
        P_fast = _bench_fast(n, T=1000, runs=20)
        t_fast = time.time() - t0
        speedup = t_nx / t_fast
        print(f"N={n:>4}  T=1000 runs=20   "
              f"nx={t_nx:6.2f}s   fast={t_fast:6.2f}s   "
              f"speedup={speedup:5.1f}x   "
              f"P_nx={P_nx:.3f}  P_fast={P_fast:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Equivalence tests (polarized_fast vs polarized)")
    print("=" * 60)
    # Vary parameters across the (a, S) plane to exercise different regimes
    cases = [
        dict(n=50,  k=8, T=50,  S=5.0,  a=0.05, init_seed=1,  sim_seed=10, graph_seed=42),
        dict(n=80,  k=8, T=100, S=20.0, a=0.10, init_seed=2,  sim_seed=11, graph_seed=43),
        dict(n=100, k=8, T=50,  S=2.0,  a=0.30, init_seed=3,  sim_seed=12, graph_seed=44),
        dict(n=100, k=8, T=50,  S=50.0, a=0.01, init_seed=4,  sim_seed=13, graph_seed=45),
        dict(n=120, k=8, T=80,  S=10.0, a=0.20, init_seed=5,  sim_seed=14, graph_seed=46),
    ]
    all_pass = True
    for c in cases:
        passed, *_ = test_equivalence(**c)
        all_pass = all_pass and passed
    print()
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    benchmark()
