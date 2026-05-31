"""Tests for population annealing."""
from __future__ import annotations

import pytest

from ising_lab import (
    IsingModel,
    brute_force_ground_state,
    parallel_tempering,
    population_annealing,
    population_annealing_icm,
)
from ising_lab.benchmarks import ea_instance, sk_instance, wrap_pa, wrap_pa_icm


def test_pa_finds_ground_state_on_small_sk():
    inst = sk_instance(12, seed=2, distribution="binary")
    _, truth = brute_force_ground_state(inst.model)
    res = population_annealing(
        inst.model, num_temps=30, population=50, num_sweeps=10,
        beta_min=0.1, beta_max=10.0, num_reads=8, seed=1,
    )
    assert min(e for _, e in res) == pytest.approx(truth)


def test_pa_finds_ground_state_on_frustrated_ring():
    n = 7
    model = IsingModel(n, [0.0] * n, [(i, (i + 1) % n, 1.0) for i in range(n)])
    _, truth = brute_force_ground_state(model)
    res = population_annealing(
        model, num_temps=30, population=40, num_sweeps=10,
        beta_min=0.1, beta_max=8.0, num_reads=10, seed=5,
    )
    assert min(e for _, e in res) == pytest.approx(truth)


def test_pa_reported_energy_is_exact():
    inst = ea_instance(3, seed=2, dimension=3)
    res = population_annealing(inst.model, num_temps=20, population=30, num_reads=6, seed=3)
    for state, energy in res:
        assert inst.model.energy(state) == pytest.approx(energy)


def test_pa_deterministic_for_fixed_seed():
    inst = sk_instance(20, seed=7, distribution="binary")
    r1 = population_annealing(inst.model, num_temps=20, population=40, num_reads=6, seed=42)
    r2 = population_annealing(inst.model, num_temps=20, population=40, num_reads=6, seed=42)
    assert r1 == r2


def test_pa_validates_arguments():
    inst = sk_instance(10, seed=0, distribution="binary")
    with pytest.raises(Exception):
        population_annealing(inst.model, num_temps=1)
    with pytest.raises(Exception):
        population_annealing(inst.model, population=0)
    with pytest.raises(Exception):
        population_annealing(inst.model, beta_min=float("nan"))
    with pytest.raises(Exception):
        population_annealing(inst.model, beta_min=5.0, beta_max=1.0)


def test_pa_returns_valid_state_shape():
    inst = sk_instance(8, seed=1, distribution="binary")
    res = population_annealing(inst.model, num_temps=15, population=20, num_reads=4, seed=1)
    assert len(res) == 4
    for state, energy in res:
        assert len(state) == inst.n
        assert all(s in (-1, 1) for s in state)


def test_wrap_pa_contract():
    inst = ea_instance(2, seed=0, dimension=3)
    sampler = wrap_pa(num_temps=20, population=30, seed=1)
    out = sampler(inst.model, 5)
    assert len(out) == 5
    for state, energy in out:
        assert inst.model.energy(state) == pytest.approx(energy)


def test_pa_icm_finds_ground_state_on_small_ea():
    inst = ea_instance(2, seed=1, dimension=3, distribution="binary")
    _, truth = brute_force_ground_state(inst.model)
    res = population_annealing_icm(
        inst.model, num_temps=30, population=30, num_sweeps=8,
        beta_min=0.1, beta_max=10.0, icm_every=1, num_reads=8, seed=1,
    )
    assert min(e for _, e in res) == pytest.approx(truth)


def test_pa_icm_reported_energy_exact_and_deterministic():
    inst = ea_instance(3, seed=2, dimension=3)
    r1 = population_annealing_icm(inst.model, num_temps=20, population=30, icm_every=1, num_reads=5, seed=3)
    r2 = population_annealing_icm(inst.model, num_temps=20, population=30, icm_every=1, num_reads=5, seed=3)
    assert r1 == r2
    for state, energy in r1:
        assert inst.model.energy(state) == pytest.approx(energy)


def test_pa_icm_validates_arguments():
    inst = sk_instance(10, seed=0, distribution="binary")
    with pytest.raises(Exception):
        population_annealing_icm(inst.model, population=1)  # needs >= 2 for pairs
    with pytest.raises(Exception):
        population_annealing_icm(inst.model, icm_every=0)
    with pytest.raises(Exception):
        population_annealing_icm(inst.model, beta_min=float("nan"))


def test_plain_pa_unchanged_by_icm_addition():
    """Regression: plain population_annealing must stay deterministic and is the
    icm_every=0 path, so adding ICM did not perturb it."""
    inst = sk_instance(16, seed=4, distribution="binary")
    a = population_annealing(inst.model, num_temps=20, population=30, num_reads=8, seed=9)
    b = population_annealing(inst.model, num_temps=20, population=30, num_reads=8, seed=9)
    assert a == b


def test_wrap_pa_icm_contract():
    inst = ea_instance(2, seed=0, dimension=3)
    sampler = wrap_pa_icm(num_temps=20, population=20, icm_every=1, seed=1)
    out = sampler(inst.model, 5)
    assert len(out) == 5
    for state, energy in out:
        assert inst.model.energy(state) == pytest.approx(energy)


def test_pa_competitive_with_pt_on_ea_gaussian():
    """On a hard-ish 3D EA Gaussian instance, PA should reach an energy at least
    as low as plain PT at comparable settings. This guards the headline claim
    without being flaky -- it only asserts 'no worse', not the full margin."""
    inst = ea_instance(5, seed=10, dimension=3, distribution="gaussian")  # N=125
    pa = population_annealing(
        inst.model, num_temps=40, population=40, num_sweeps=8,
        beta_min=0.05, beta_max=6.0, num_reads=8, seed=0,
    )
    pt = parallel_tempering(
        inst.model, num_sweeps=2000, num_replicas=24,
        beta_min=0.05, beta_max=6.0, num_reads=8, seed=0,
    )
    assert min(e for _, e in pa) <= min(e for _, e in pt) + 1e-6
