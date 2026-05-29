"""End-to-end smoke tests for the Ising lab."""
from __future__ import annotations

import itertools

import pytest

from ising_lab import IsingModel, QUBO, qubo_to_ising, simulated_anneal, parallel_tempering
from ising_lab.problems import (
    graph_coloring,
    graph_coloring_decode,
    knapsack,
    knapsack_decode,
    max_cut,
    number_partition,
    tour_cost,
    tsp,
    tsp_decode_tour,
)
from ising_lab.qubo import spins_to_bits


def brute_force_min_energy(model: IsingModel, offset: float = 0.0) -> float:
    """Exhaustive ground-state search for small models, for use as oracle."""
    n = model.num_spins
    best = float("inf")
    for bits in itertools.product((-1, 1), repeat=n):
        e = model.energy(list(bits)) + offset
        if e < best:
            best = e
    return best


def test_qubo_energy_matches_ising_energy():
    """For every assignment, QUBO energy and (Ising energy + offset) must agree."""
    q = QUBO(num_vars=3, offset=0.5)
    q.add_linear(0, 1.0)
    q.add_linear(1, -2.0)
    q.add_quadratic(0, 1, 3.0)
    q.add_quadratic(1, 2, -1.5)

    model, offset = qubo_to_ising(q)
    for bits in itertools.product((0, 1), repeat=3):
        spins = [2 * b - 1 for b in bits]
        assert q.energy(bits) == pytest.approx(model.energy(spins) + offset)


def test_simulated_anneal_finds_ground_state_on_triangle():
    """Ferromagnetic triangle: ground states are all-up and all-down at E = -3."""
    model = IsingModel(3, [0.0, 0.0, 0.0], [(0, 1, -1.0), (0, 2, -1.0), (1, 2, -1.0)])
    results = simulated_anneal(model, num_sweeps=500, num_reads=10, seed=42)
    best = min(e for _, e in results)
    assert best == pytest.approx(-3.0)


def test_max_cut_on_square_finds_optimal_cut():
    """4-cycle has max-cut 4 (bipartition 0,2 vs 1,3)."""
    model, offset = max_cut(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    results = simulated_anneal(model, num_sweeps=500, num_reads=20, seed=7)
    best_state, best_e = min(results, key=lambda r: r[1])
    cut_size = -(best_e + offset)
    assert cut_size == pytest.approx(4.0)
    bits = spins_to_bits(best_state)
    # Verify the bipartition really cuts every edge.
    cut_check = sum(1 for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)] if bits[i] != bits[j])
    assert cut_check == 4


def test_number_partition_perfect_split():
    """{3, 1, 1, 2, 2, 1} has a perfect partition (sum 5 each), so ground energy is 0."""
    nums = [3.0, 1.0, 1.0, 2.0, 2.0, 1.0]
    model, offset = number_partition(nums)
    results = simulated_anneal(model, num_sweeps=2000, num_reads=30, seed=123)
    best_e = min(e for _, e in results) + offset
    assert best_e == pytest.approx(0.0, abs=1e-9)


def test_tsp_square_finds_perimeter_tour():
    """4 cities on a unit square. Optimal tour is the perimeter with cost 4.0;
    the crossing tour 0-2-1-3 costs 2 + 2*sqrt(2) ~ 4.83."""
    import math

    sqrt2 = math.sqrt(2.0)
    d = [
        [0.0, 1.0, sqrt2, 1.0],
        [1.0, 0.0, 1.0, sqrt2],
        [sqrt2, 1.0, 0.0, 1.0],
        [1.0, sqrt2, 1.0, 0.0],
    ]
    model, offset = tsp(d)
    results = simulated_anneal(
        model, num_sweeps=3000, num_reads=40, beta_start=0.05, beta_end=20.0, seed=2024
    )
    best_state, _ = min(results, key=lambda r: r[1])
    tour = tsp_decode_tour(best_state, num_cities=4)
    assert sorted(tour) == [0, 1, 2, 3], f"tour {tour} is not a valid permutation"
    assert tour_cost(tour, d) == pytest.approx(4.0)


def test_brute_force_matches_anneal_on_small_random_model():
    """SA should at least match the brute-force optimum on a tiny instance."""
    model = IsingModel(
        5,
        [0.3, -0.4, 0.1, 0.0, -0.2],
        [(0, 1, 0.5), (1, 2, -0.7), (2, 3, 0.4), (3, 4, -0.3), (0, 4, 0.6)],
    )
    truth = brute_force_min_energy(model)
    results = simulated_anneal(model, num_sweeps=1000, num_reads=20, seed=99)
    best = min(e for _, e in results)
    assert best == pytest.approx(truth)


def test_parallel_tempering_finds_ground_state_on_frustrated_ring():
    """Antiferromagnetic odd ring is geometrically frustrated; PT should still find the optimum."""
    # 7-spin antiferromagnetic ring: J = +1 on every edge, no field.
    n = 7
    edges = [(i, (i + 1) % n) for i in range(n)]
    model = IsingModel(n, [0.0] * n, [(min(i, j), max(i, j), 1.0) for i, j in edges])
    truth = brute_force_min_energy(model)
    results = parallel_tempering(
        model,
        num_sweeps=400,
        num_replicas=8,
        beta_min=0.1,
        beta_max=8.0,
        swap_every=1,
        num_reads=10,
        seed=11,
    )
    best = min(e for _, e in results)
    assert best == pytest.approx(truth)


def test_graph_coloring_3_color_pentagon():
    """C5 (odd cycle) has chromatic number 3. Need K=3 to find a valid coloring."""
    n = 5
    edges = [(i, (i + 1) % n) for i in range(n)]
    model, offset = graph_coloring(n, edges, num_colors=3)
    results = parallel_tempering(
        model,
        num_sweeps=400,
        num_replicas=6,
        beta_min=0.1,
        beta_max=10.0,
        num_reads=10,
        seed=2026,
    )
    best_state, best_e = min(results, key=lambda r: r[1])
    assert best_e + offset == pytest.approx(0.0, abs=1e-9), "ground state should be feasible"
    colors = graph_coloring_decode(best_state, num_nodes=n, num_colors=3)
    for u, v in edges:
        assert colors[u] != colors[v], f"edge ({u},{v}) has matching colors {colors[u]}"


def test_knapsack_finds_optimal_selection():
    """Knapsack with weights [3,4,5], values [4,5,6], capacity 8. Optimum is items {0,2}."""
    weights = [3, 4, 5]
    values = [4.0, 5.0, 6.0]
    capacity = 8

    model, offset = knapsack(weights, values, capacity)
    results = parallel_tempering(
        model,
        num_sweeps=500,
        num_replicas=8,
        beta_min=0.05,
        beta_max=15.0,
        num_reads=20,
        seed=2027,
    )
    best_state, _ = min(results, key=lambda r: r[1])
    items = knapsack_decode(best_state, num_items=3)
    assert sorted(items) == [0, 2]
    assert sum(weights[i] for i in items) <= capacity
    assert sum(values[i] for i in items) == pytest.approx(10.0)


def test_knapsack_capacity_zero_forces_empty_selection():
    """Capacity 0 means nothing fits; optimal selection must be empty."""
    model, offset = knapsack([1, 2, 3], [5.0, 5.0, 5.0], capacity=0)
    results = parallel_tempering(model, num_sweeps=200, num_reads=5, seed=4)
    best_state, _ = min(results, key=lambda r: r[1])
    items = knapsack_decode(best_state, num_items=3)
    assert items == []


def test_graph_coloring_2_colors_infeasible_on_triangle():
    """K3 is not 2-colorable; the ground state must have positive energy."""
    model, offset = graph_coloring(3, [(0, 1), (1, 2), (0, 2)], num_colors=2)
    results = parallel_tempering(model, num_sweeps=200, num_reads=5, seed=3)
    best_e = min(e for _, e in results) + offset
    assert best_e > 0.0, "no valid 2-coloring should exist for a triangle"


def test_sa_results_deterministic_for_fixed_seed():
    """Despite running reads in parallel, same seed must give identical results."""
    model = IsingModel(
        6,
        [0.3, -0.4, 0.1, 0.0, -0.2, 0.5],
        [(0, 1, 0.5), (1, 2, -0.7), (2, 3, 0.4), (3, 4, -0.3), (4, 5, 0.6)],
    )
    r1 = simulated_anneal(model, num_sweeps=200, num_reads=16, seed=2026)
    r2 = simulated_anneal(model, num_sweeps=200, num_reads=16, seed=2026)
    assert r1 == r2


def test_pt_results_deterministic_for_fixed_seed():
    """Parallelizing PT chains must not change per-seed output."""
    model = IsingModel(
        7,
        [0.0] * 7,
        [(i, (i + 1) % 7, 1.0) for i in range(7)],
    )
    # Fix only-upper indices for IsingModel constructor.
    fixed = [(min(i, j), max(i, j), w) for i, j, w in [(i, (i + 1) % 7, 1.0) for i in range(7)]]
    model = IsingModel(7, [0.0] * 7, fixed)
    r1 = parallel_tempering(model, num_sweeps=200, num_reads=12, seed=99)
    r2 = parallel_tempering(model, num_sweeps=200, num_reads=12, seed=99)
    assert r1 == r2


def test_parallel_tempering_returns_valid_state_shape():
    """Sanity: returned states must be num_reads x num_spins with entries in {-1, +1}."""
    model = IsingModel(4, [0.1, -0.2, 0.0, 0.3], [(0, 1, 0.5), (2, 3, -0.4)])
    results = parallel_tempering(
        model, num_sweeps=100, num_replicas=4, num_reads=3, seed=1
    )
    assert len(results) == 3
    for state, energy in results:
        assert len(state) == 4
        assert all(s in (-1, 1) for s in state)
        assert model.energy(state) == pytest.approx(energy)
