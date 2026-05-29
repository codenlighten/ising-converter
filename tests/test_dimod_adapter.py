"""Tests for the dimod adapter: BQM <-> IsingModel conversion and samplers."""
from __future__ import annotations

import pytest

dimod = pytest.importorskip("dimod")

from ising_lab import IsingModel
from ising_lab.dimod_adapter import (
    ParallelTemperingSampler,
    SimulatedAnnealingSampler,
    from_bqm,
    to_bqm,
)


def test_from_bqm_spin_preserves_energy():
    """Ising energy + offset must equal BQM energy for every assignment."""
    bqm = dimod.BinaryQuadraticModel(
        {"a": 0.5, "b": -0.3, "c": 0.2},
        {("a", "b"): 1.0, ("b", "c"): -0.4, ("a", "c"): 0.6},
        offset=0.7,
        vartype="SPIN",
    )
    model, offset, labels = from_bqm(bqm)
    assert labels == ["a", "b", "c"]
    for s in ((-1, -1, -1), (-1, 1, -1), (1, 1, 1), (1, -1, 1)):
        sample = dict(zip(labels, s))
        assert bqm.energy(sample) == pytest.approx(model.energy(list(s)) + offset)


def test_from_bqm_binary_preserves_energy():
    """BINARY BQM energy must match Ising energy + offset under x = (1+s)/2."""
    bqm = dimod.BinaryQuadraticModel(
        {0: 1.0, 1: -2.0},
        {(0, 1): 3.0},
        offset=0.25,
        vartype="BINARY",
    )
    model, offset, labels = from_bqm(bqm)
    for bits in ((0, 0), (0, 1), (1, 0), (1, 1)):
        spins = [2 * b - 1 for b in bits]
        sample = dict(zip(labels, bits))
        assert bqm.energy(sample) == pytest.approx(model.energy(spins) + offset)


def test_to_bqm_roundtrip_preserves_energy():
    """IsingModel -> BQM -> energies must match the underlying model + offset."""
    model = IsingModel(3, [0.3, -0.1, 0.4], [(0, 1, 0.5), (1, 2, -0.6)])
    bqm = to_bqm(model, offset=0.2, labels=["x", "y", "z"])
    assert set(bqm.variables) == {"x", "y", "z"}
    for s in ((-1, -1, -1), (1, 1, 1), (1, -1, 1), (-1, 1, -1)):
        sample = dict(zip(["x", "y", "z"], s))
        assert bqm.energy(sample) == pytest.approx(model.energy(list(s)) + 0.2)


def test_sa_sampler_energies_match_bqm():
    """SimulatedAnnealingSampler must report energies consistent with bqm.energy."""
    bqm = dimod.BinaryQuadraticModel(
        {0: 0.5, 1: -0.5, 2: 0.0},
        {(0, 1): 1.0, (1, 2): -1.0},
        offset=0.0,
        vartype="SPIN",
    )
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_sweeps=200, num_reads=8, seed=42)
    assert len(sampleset) == 8
    for record in sampleset.record:
        sample = dict(zip(sampleset.variables, record.sample))
        assert bqm.energy(sample) == pytest.approx(record.energy)


def test_pt_sampler_finds_ground_state_on_binary_bqm():
    """PT on a BINARY BQM must hit the same minimum dimod's ExactSolver finds."""
    bqm = dimod.BinaryQuadraticModel(
        {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0},
        {(0, 1): 2.0, (1, 2): 2.0, (2, 3): 2.0, (0, 3): 2.0},
        offset=0.0,
        vartype="BINARY",
    )
    truth = dimod.ExactSolver().sample(bqm).first.energy

    sampler = ParallelTemperingSampler()
    sampleset = sampler.sample(
        bqm,
        num_sweeps=300,
        num_replicas=6,
        beta_min=0.1,
        beta_max=8.0,
        num_reads=10,
        seed=7,
    )
    assert sampleset.first.energy == pytest.approx(truth)
    # sampleset.first.sample is already a Mapping {var: value}; bqm.energy accepts it.
    assert bqm.energy(sampleset.first.sample) == pytest.approx(truth)
