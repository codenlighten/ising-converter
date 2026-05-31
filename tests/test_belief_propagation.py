"""Tests for belief propagation: tree-exactness, Bethe free energy, and the
honest characterization of its ground-state heuristic."""
from __future__ import annotations

import itertools
import math

import pytest

from ising_lab import IsingModel, belief_propagation, bp_ground_state, bp_marginals
from ising_lab.inference import _simple_cycles_up_to, loop_corrected_free_energy


def _exact_marginals_and_free_energy(model: IsingModel, beta: float):
    n = model.num_spins
    z = 0.0
    mz = [0.0] * n
    for bits in itertools.product((-1, 1), repeat=n):
        w = math.exp(-beta * model.energy(list(bits)))
        z += w
        for i in range(n):
            mz[i] += w * bits[i]
    return [m / z for m in mz], -math.log(z) / beta


def _model(n, edges, h=None):
    return IsingModel(n, h or [0.0] * n, [(min(i, j), max(i, j), w) for i, j, w in edges])


def test_bp_exact_on_path_tree():
    """BP marginals and Bethe free energy are exact on a tree (a path graph)."""
    beta = 0.8
    m = _model(5, [(0, 1, -1.0), (1, 2, 0.7), (2, 3, -0.4), (3, 4, 1.2)],
               h=[0.3, -0.5, 0.2, 0.0, -0.1])
    marg, free_energy, converged, _ = belief_propagation(m, beta, damping=0.0, tol=1e-12)
    em, ef = _exact_marginals_and_free_energy(m, beta)
    assert converged
    assert max(abs(a - b) for a, b in zip(marg, em)) < 1e-9
    assert free_energy == pytest.approx(ef, abs=1e-9)


def test_bp_exact_on_star_tree():
    beta = 0.8
    m = _model(5, [(0, 1, 0.6), (0, 2, -0.9), (0, 3, 0.4), (0, 4, -0.5)],
               h=[-0.2, 0.1, 0.3, -0.4, 0.5])
    marg, free_energy, converged, _ = belief_propagation(m, beta, damping=0.0, tol=1e-12)
    em, ef = _exact_marginals_and_free_energy(m, beta)
    assert max(abs(a - b) for a, b in zip(marg, em)) < 1e-9
    assert free_energy == pytest.approx(ef, abs=1e-9)


def test_bp_single_edge_free_energy_is_exact():
    beta = 1.3
    m = _model(2, [(0, 1, -0.7)], h=[0.4, -0.2])
    _, free_energy, _, _ = belief_propagation(m, beta, damping=0.0, tol=1e-12)
    _, ef = _exact_marginals_and_free_energy(m, beta)
    assert free_energy == pytest.approx(ef, abs=1e-12)


def test_bp_validates_arguments():
    m = _model(3, [(0, 1, 1.0), (1, 2, 1.0)])
    with pytest.raises(Exception):
        belief_propagation(m, beta=0.0)
    with pytest.raises(Exception):
        belief_propagation(m, beta=-1.0)
    with pytest.raises(Exception):
        belief_propagation(m, beta=1.0, damping=1.0)
    with pytest.raises(Exception):
        belief_propagation(m, beta=1.0, damping=-0.1)


def test_bp_ground_state_on_ferromagnetic_tree():
    """On an unfrustrated tree with a symmetry-breaking field, rounding high-beta
    BP marginals recovers the exact ground state (BP is exact on trees)."""
    # Ferromagnetic path (J < 0 aligns spins) with a small positive field.
    m = _model(8, [(i, i + 1, -1.0) for i in range(7)], h=[0.1] * 8)
    n = m.num_spins
    truth = min(m.energy(list(b)) for b in itertools.product((-1, 1), repeat=n))
    state, energy, converged = bp_ground_state(m, beta=3.0)
    assert converged
    assert energy == pytest.approx(truth)


def test_bp_marginals_in_range_and_convergence_flag():
    m = _model(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])  # antiferro chain (tree)
    marg, _, converged, iters = bp_marginals(m, beta=1.0, damping=0.0, tol=1e-12)
    assert all(-1.0 <= x <= 1.0 for x in marg)
    assert converged
    assert iters >= 1


def test_bp_ground_state_returns_convergence_flag_on_loopy_graph():
    """On a frustrated loopy graph BP is only a heuristic; the call must still
    return a valid state and a convergence flag the caller can check."""
    # Frustrated triangle: no field, fully symmetric.
    m = _model(3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)])
    state, energy, converged = bp_ground_state(m, beta=2.0, damping=0.5)
    assert len(state) == 3
    assert all(s in (-1, 1) for s in state)
    assert energy == pytest.approx(m.energy(state))
    assert isinstance(converged, bool)


# ---------- Loop-corrected BP (Chertkov-Chernyak) ----------


def test_simple_cycle_enumerator_counts_k4():
    """K4 has 4 triangles + 3 four-cycles = 7 simple cycles, each counted once."""
    adj = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    cycles = _simple_cycles_up_to(4, adj, max_len=4)
    assert len(cycles) == 7
    assert sum(1 for c in cycles if len(c) == 3) == 4
    assert sum(1 for c in cycles if len(c) == 4) == 3
    # Length bound is respected.
    assert len(_simple_cycles_up_to(4, adj, max_len=3)) == 4


def test_loop_correction_is_exact_on_a_ring():
    """A ring is a single cycle, so the loop series terminates: loop-corrected
    free energy must equal the exact free energy."""
    beta = 0.6
    n = 6
    ring = _model(n, [(i, (i + 1) % n, 0.8) for i in range(n)], h=[0.3] * n)
    exact, _ = _exact_marginals_and_free_energy(ring, beta)  # marginals unused
    exact_f = _exact_marginals_and_free_energy(ring, beta)[1]
    loop_f, bethe_f, n_cycles, converged = loop_corrected_free_energy(
        ring, beta, max_cycle_len=n
    )
    assert converged
    assert n_cycles == 1
    assert loop_f == pytest.approx(exact_f, abs=1e-9)
    # And it genuinely corrected the Bethe value.
    assert abs(loop_f - exact_f) < abs(bethe_f - exact_f)


def test_loop_correction_improves_free_energy_on_loopy_grid():
    """On a frustrated 3x3 grid the truncated loop correction must be strictly
    closer to the exact free energy than the plain Bethe value."""
    beta = 0.6
    # Fixed frustrated 3x3 grid (mixed +/-1 couplings) with small fields.
    edges = [
        (0, 1, 1.0), (1, 2, -1.0), (3, 4, -1.0), (4, 5, 1.0), (6, 7, 1.0), (7, 8, -1.0),
        (0, 3, -1.0), (3, 6, 1.0), (1, 4, 1.0), (4, 7, -1.0), (2, 5, -1.0), (5, 8, 1.0),
    ]
    m = _model(9, edges, h=[0.1, -0.2, 0.15, 0.0, -0.1, 0.2, -0.05, 0.1, -0.15])
    exact_f = _exact_marginals_and_free_energy(m, beta)[1]
    loop_f, bethe_f, n_cycles, converged = loop_corrected_free_energy(m, beta, max_cycle_len=8)
    assert converged
    assert n_cycles >= 1
    assert abs(loop_f - exact_f) < abs(bethe_f - exact_f)


def test_loop_correction_validates_cycle_length():
    m = _model(3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)])
    with pytest.raises(ValueError):
        loop_corrected_free_energy(m, beta=0.5, max_cycle_len=2)
