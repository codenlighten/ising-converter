"""Tests for the Edwards-Anderson lattice spin-glass generator."""
from __future__ import annotations

import pytest

from ising_lab.benchmarks import ea_instance, ea_suite


def test_ea_3d_periodic_has_3N_couplings():
    """3D cubic lattice with periodic boundaries: each site has 6 neighbors,
    so total edges = 6*N/2 = 3*N."""
    L = 4
    inst = ea_instance(L, seed=0, dimension=3, periodic=True)
    assert inst.n == L ** 3
    assert len(inst.model.couplings()) == 3 * inst.n


def test_ea_2d_periodic_has_2N_couplings():
    """2D square lattice with periodic boundaries: each site has 4 neighbors,
    edges = 4*N/2 = 2*N."""
    L = 5
    inst = ea_instance(L, seed=0, dimension=2, periodic=True)
    assert inst.n == L ** 2
    assert len(inst.model.couplings()) == 2 * inst.n


def test_ea_3d_open_has_fewer_couplings():
    """Open boundaries reduce edge count vs periodic."""
    L = 4
    periodic = ea_instance(L, seed=0, dimension=3, periodic=True)
    open_ = ea_instance(L, seed=0, dimension=3, periodic=False)
    assert len(open_.model.couplings()) < len(periodic.model.couplings())


def test_ea_distribution_key_uniqueness():
    """Different (dim, dist, L) produce different distribution tags so the
    registry can keep EA and SK instances apart."""
    ea2 = ea_instance(3, seed=0, dimension=2, distribution="binary")
    ea3 = ea_instance(3, seed=0, dimension=3, distribution="binary")
    ea3g = ea_instance(3, seed=0, dimension=3, distribution="gaussian")
    assert ea2.distribution != ea3.distribution
    assert ea3.distribution != ea3g.distribution
    assert "ea-2d-binary-L3" == ea2.distribution
    assert "ea-3d-gaussian-L3" == ea3g.distribution


def test_ea_reproducibility_across_seeds():
    inst1 = ea_instance(4, seed=42, dimension=3)
    inst2 = ea_instance(4, seed=42, dimension=3)
    assert inst1.model.couplings() == inst2.model.couplings()


def test_ea_suite_size():
    suite = ea_suite([3, 4], instances_per_L=2, dimension=3)
    assert len(suite) == 4
    seeds = [s.seed for s in suite]
    assert len(set(seeds)) == 4


def test_ea_binary_coupling_values():
    """Binary distribution must produce strictly +/- 1 weights."""
    inst = ea_instance(4, seed=0, dimension=3, distribution="binary")
    for _, _, w in inst.model.couplings():
        assert w in (-1.0, 1.0)
