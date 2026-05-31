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

from typing import List, Optional, Tuple

from ._kernel import IsingModel, belief_propagation


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


__all__ = ["bp_marginals", "bp_ground_state"]
