"""Belief-propagation inference helpers.

Belief propagation is a deterministic message-passing alternative to the Monte
Carlo samplers. It computes spin marginals and the Bethe free energy -- exact on
trees, approximate (the Bethe approximation) on loopy graphs. These are
observables the MC methods do not directly produce.

For *optimization*, BP is weaker than the samplers on frustrated loopy graphs
(it may not converge, and rounding its marginals need not give the ground
state); `bp_ground_state` is offered as a fast deterministic baseline and a
warm-start source, not as a competitor to population annealing on 3D glasses.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from ._kernel import IsingModel, belief_propagation, bp_correlations


def bp_marginals(
    model: IsingModel,
    beta: float,
    max_iters: int = 1000,
    damping: float = 0.5,
    tol: float = 1e-8,
) -> Tuple[List[float], float, bool, int]:
    """Run sum-product BP. Returns (marginals, bethe_free_energy, converged, iters).

    `marginals[i]` is the BP estimate of <s_i> in [-1, 1]. On a tree the result
    is exact; inspect `converged` before trusting it on a loopy graph.
    """
    return belief_propagation(model, beta, max_iters=max_iters, damping=damping, tol=tol)


def bp_ground_state(
    model: IsingModel,
    beta: float = 3.0,
    max_iters: int = 2000,
    damping: float = 0.5,
    tol: float = 1e-8,
) -> Tuple[List[int], float, bool]:
    """Estimate a ground state by rounding high-beta BP marginals to spins.

    Returns (state, energy, converged). Deterministic and fast, but only a
    heuristic on loopy frustrated graphs -- `converged=False` means BP did not
    reach a fixed point and the estimate should be treated with suspicion.
    """
    marginals, _, converged, _ = belief_propagation(
        model, beta, max_iters=max_iters, damping=damping, tol=tol
    )
    state = [1 if m >= 0.0 else -1 for m in marginals]
    return state, model.energy(state), converged


def _simple_cycles_up_to(n: int, adj: List[List[int]], max_len: int) -> List[List[int]]:
    """Enumerate every simple cycle (length 3..max_len) once.

    Canonical form: each cycle is rooted at its smallest vertex `start`, only
    vertices >= start are visited, and the direction is fixed by requiring the
    second vertex to be smaller than the last -- so a cycle and its reverse are
    counted a single time.
    """
    cycles: List[List[int]] = []

    def dfs(start: int, v: int, path: List[int], on_path: set) -> None:
        for u in adj[v]:
            if u < start:
                continue
            if u == start:
                if len(path) >= 3 and path[1] < path[-1]:
                    cycles.append(path[:])
            elif u not in on_path and len(path) < max_len:
                on_path.add(u)
                path.append(u)
                dfs(start, u, path, on_path)
                path.pop()
                on_path.remove(u)

    for start in range(n):
        dfs(start, start, [start], {start})
    return cycles


def loop_corrected_free_energy(
    model: IsingModel,
    beta: float,
    max_cycle_len: int = 8,
    max_iters: int = 2000,
    damping: float = 0.5,
    tol: float = 1e-10,
) -> Tuple[float, float, int, bool]:
    """Chertkov-Chernyak loop-corrected Bethe free energy.

    The exact partition function is Z = Z_Bethe * (1 + sum over generalized
    loops). Truncating the sum at simple cycles gives the leading correction:

        r(C) = prod_{(ij) in C} chi_ij / sqrt((1 - m_i^2)(1 - m_j^2)),

    where chi_ij is the BP connected correlation and m_i the BP marginal. This
    is *exact* on a single cycle (a ring) and a systematic improvement over the
    plain Bethe free energy on loopy graphs; the truncation error grows with
    graph density (higher-order generalized loops are dropped).

    Returns (loop_corrected_F, bethe_F, num_cycles, converged). Inspect
    `converged`: if BP did not reach a fixed point the correction is unreliable.
    """
    if max_cycle_len < 3:
        raise ValueError("max_cycle_len must be >= 3")
    marginals, edge_i, edge_j, chi, bethe_f, converged, _ = bp_correlations(
        model, beta, max_iters=max_iters, damping=damping, tol=tol
    )
    n = model.num_spins
    adj: List[List[int]] = [[] for _ in range(n)]
    chi_map = {}
    for a, b, c in zip(edge_i, edge_j, chi):
        adj[a].append(b)
        adj[b].append(a)
        chi_map[(a, b)] = c

    cycle_sum = 0.0
    cycles = _simple_cycles_up_to(n, adj, max_cycle_len)
    for cyc in cycles:
        r = 1.0
        for idx in range(len(cyc)):
            i, j = cyc[idx], cyc[(idx + 1) % len(cyc)]
            key = (i, j) if i < j else (j, i)
            denom = math.sqrt((1.0 - marginals[i] ** 2) * (1.0 - marginals[j] ** 2))
            r *= chi_map[key] / denom if denom > 0.0 else 0.0
        cycle_sum += r

    z_ratio = 1.0 + cycle_sum
    if z_ratio <= 0.0:
        # Correction drove the partition-function ratio non-positive (the
        # truncation broke down); fall back to the plain Bethe value.
        return bethe_f, bethe_f, len(cycles), converged
    loop_f = bethe_f - math.log(z_ratio) / beta
    return loop_f, bethe_f, len(cycles), converged


__all__ = [
    "bp_marginals",
    "bp_ground_state",
    "loop_corrected_free_energy",
]
