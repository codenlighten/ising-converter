"""Tests for Houdayer isoenergetic-cluster-move parallel tempering and the
Parisi SK energy-density reference."""
from __future__ import annotations

import pytest

from ising_lab import (
    IsingModel,
    brute_force_ground_state,
    parallel_tempering,
    parallel_tempering_houdayer,
)
from ising_lab.benchmarks import (
    PARISI_SK_ENERGY_DENSITY,
    ea_instance,
    sk_energy_density,
    sk_instance,
    sk_parisi_reference_energy,
    wrap_pt_houdayer,
)


# ---------- Houdayer-PT kernel ----------


def test_houdayer_finds_ground_state_on_small_ea_3d():
    """On a brute-forceable 3D EA cube (L=2, N=8), Houdayer-PT must hit the
    exact ground state."""
    inst = ea_instance(2, seed=1, dimension=3, distribution="binary")
    _, truth = brute_force_ground_state(inst.model)
    res = parallel_tempering_houdayer(
        inst.model, num_sweeps=500, num_replicas=6,
        beta_min=0.1, beta_max=10.0, icm_every=5, num_reads=8, seed=1,
    )
    assert min(e for _, e in res) == pytest.approx(truth)


def test_houdayer_finds_ground_state_on_frustrated_ring():
    """Antiferromagnetic odd ring (sparse, frustrated) -- Houdayer-PT optimum
    must match brute force."""
    n = 7
    model = IsingModel(n, [0.0] * n, [(i, (i + 1) % n, 1.0) for i in range(n)])
    _, truth = brute_force_ground_state(model)
    res = parallel_tempering_houdayer(
        model, num_sweeps=400, num_replicas=8, beta_min=0.1, beta_max=8.0,
        icm_every=3, num_reads=10, seed=11,
    )
    assert min(e for _, e in res) == pytest.approx(truth)


def test_houdayer_reported_energy_is_exact():
    """Returned energy must equal model.energy(state) exactly (no drift)."""
    inst = ea_instance(3, seed=2, dimension=3)
    res = parallel_tempering_houdayer(
        inst.model, num_sweeps=300, num_replicas=8, num_reads=6, seed=3,
    )
    for state, energy in res:
        assert inst.model.energy(state) == pytest.approx(energy)


def test_houdayer_deterministic_for_fixed_seed():
    """Two-lane ICM + parallel reads must still be reproducible per seed."""
    inst = ea_instance(3, seed=7, dimension=3)
    r1 = parallel_tempering_houdayer(inst.model, num_sweeps=300, num_replicas=8, num_reads=6, seed=42)
    r2 = parallel_tempering_houdayer(inst.model, num_sweeps=300, num_replicas=8, num_reads=6, seed=42)
    assert r1 == r2


def test_houdayer_validates_arguments():
    inst = ea_instance(2, seed=0, dimension=3)
    with pytest.raises(Exception):
        parallel_tempering_houdayer(inst.model, num_replicas=1)
    with pytest.raises(Exception):
        parallel_tempering_houdayer(inst.model, beta_min=float("nan"))
    with pytest.raises(Exception):
        parallel_tempering_houdayer(inst.model, icm_every=0)
    with pytest.raises(Exception):
        parallel_tempering_houdayer(inst.model, swap_every=0)


def test_houdayer_is_noop_on_fully_connected_sk():
    """On fully connected SK the disagreement graph percolates into a single
    cluster, so ICM degenerates to a global swap. Houdayer-PT should therefore
    do no *better* than plain PT here (within sampling noise) -- this guards the
    documented regime claim. We assert it doesn't find a *lower* energy than PT
    given equal compute (PT gets 2x reads to match the two lanes)."""
    inst = sk_instance(40, seed=5, distribution="binary")
    pt = parallel_tempering(
        inst.model, num_sweeps=1500, num_replicas=12, beta_min=0.1, beta_max=8.0,
        num_reads=16, seed=0,
    )
    hou = parallel_tempering_houdayer(
        inst.model, num_sweeps=1500, num_replicas=12, beta_min=0.1, beta_max=8.0,
        icm_every=10, num_reads=8, seed=0,
    )
    # Both reach the same floor on this easy size; Houdayer must not beat PT.
    assert min(e for _, e in hou) >= min(e for _, e in pt) - 1e-6


def test_wrap_pt_houdayer_contract():
    """The wrapped sampler obeys the (model, num_reads) -> [(state, energy)] contract."""
    inst = ea_instance(2, seed=0, dimension=3)
    sampler = wrap_pt_houdayer(num_sweeps=200, num_replicas=6, icm_every=5, seed=1)
    out = sampler(inst.model, 5)
    assert len(out) == 5
    for state, energy in out:
        assert len(state) == inst.n
        assert all(s in (-1, 1) for s in state)
        assert inst.model.energy(state) == pytest.approx(energy)


# ---------- Parisi SK energy-density reference ----------


def test_parisi_density_constant_in_expected_range():
    """The Parisi SK ground-state energy density is ~-0.7632."""
    assert 0.762 < PARISI_SK_ENERGY_DENSITY < 0.764


def test_sk_energy_density_and_reference_are_consistent():
    """sk_energy_density(reference, N) recovers exactly -PARISI."""
    n = 200
    ref = sk_parisi_reference_energy(n)
    assert ref < 0
    assert sk_energy_density(ref, n) == pytest.approx(-PARISI_SK_ENERGY_DENSITY)
    # Scaling is N^{3/2}.
    assert sk_parisi_reference_energy(400) / sk_parisi_reference_energy(100) == pytest.approx(
        (400 / 100) ** 1.5
    )


def test_finite_n_sk_energy_lies_above_parisi_floor():
    """A real finite-N SK ground state sits *above* (less negative than) the
    asymptotic Parisi reference -- the reference is a thermodynamic-limit floor,
    not an exact finite-N optimum."""
    inst = sk_instance(24, seed=9, distribution="gaussian")
    _, e0 = brute_force_ground_state(inst.model)
    ref = sk_parisi_reference_energy(inst.n)
    # Exact ground state is above the asymptotic floor (finite-size correction).
    assert e0 > ref
    # But the density should be in a sane band around the Parisi value.
    assert -0.95 < sk_energy_density(e0, inst.n) < -0.55
