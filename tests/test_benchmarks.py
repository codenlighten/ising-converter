"""End-to-end test of the SK benchmark harness."""
from __future__ import annotations

import itertools

import pytest

from ising_lab import IsingModel, brute_force_min_energy
from ising_lab.benchmarks import (
    BenchmarkRecord,
    benchmark,
    sk_instance,
    sk_suite,
    solve_ground_state,
    wrap_pt,
    wrap_sa,
)


def python_brute_force(model: IsingModel) -> float:
    """Reference brute force in Python, to cross-check the Rust kernel."""
    best = float("inf")
    for bits in itertools.product((-1, 1), repeat=model.num_spins):
        e = model.energy(list(bits))
        if e < best:
            best = e
    return best


def test_sk_instance_is_reproducible_and_fully_connected():
    inst = sk_instance(8, seed=42, distribution="binary")
    assert inst.n == 8
    assert inst.seed == 42
    # SK is fully connected -- N(N-1)/2 couplings.
    assert len(inst.model.couplings()) == 8 * 7 // 2
    # All weights are +/- 1 in binary mode, no fields.
    for w in inst.model.h():
        assert w == 0.0
    for _, _, w in inst.model.couplings():
        assert w in (-1.0, 1.0)
    # Same seed -> same instance (couplings identical).
    inst2 = sk_instance(8, seed=42, distribution="binary")
    assert inst.model.couplings() == inst2.model.couplings()


def test_brute_force_matches_python_reference():
    inst = sk_instance(8, seed=7, distribution="binary")
    rust_min = brute_force_min_energy(inst.model)
    py_min = python_brute_force(inst.model)
    assert rust_min == pytest.approx(py_min)


def test_benchmark_records_have_consistent_stats():
    """Run SA + PT on one tiny SK instance and sanity-check the BenchmarkRecord fields."""
    inst = sk_instance(8, seed=123, distribution="binary")
    solve_ground_state(inst)

    samplers = {
        "sa": wrap_sa(num_sweeps=300, beta_start=0.1, beta_end=10.0, seed=1),
        "pt": wrap_pt(num_sweeps=300, num_replicas=6, beta_min=0.1, beta_max=10.0, seed=2),
    }
    records = benchmark(samplers, [inst], num_reads=20)
    assert len(records) == 2

    by_name = {r.sampler: r for r in records}
    for name in ("sa", "pt"):
        r = by_name[name]
        assert isinstance(r, BenchmarkRecord)
        assert r.n == 8
        assert r.num_reads == 20
        assert r.ground_state_energy == pytest.approx(inst.ground_state_energy)
        assert r.best_energy >= inst.ground_state_energy - 1e-9
        assert r.median_energy >= r.best_energy - 1e-9
        assert r.success_count <= r.num_reads
        assert 0.0 <= r.success_prob <= 1.0
        # PT on N=8 SK with 20 reads should find the optimum at least once.
        if name == "pt":
            assert r.success_count >= 1, f"PT failed to find ground state: {r}"


def test_tts_is_finite_when_success_probability_positive():
    """A sampler that finds the optimum sometimes should report a finite TTS_99."""
    inst = sk_instance(8, seed=5, distribution="binary")
    solve_ground_state(inst)
    samplers = {"pt": wrap_pt(num_sweeps=400, num_replicas=6, seed=11)}
    records = benchmark(samplers, [inst], num_reads=30)
    r = records[0]
    if r.success_count >= 1:
        assert r.tts_99 is not None and r.tts_99 > 0.0


def test_sk_suite_size_and_seed_uniqueness():
    suite = sk_suite([6, 8], instances_per_size=3, base_seed=100)
    assert len(suite) == 6
    seeds = [inst.seed for inst in suite]
    assert len(set(seeds)) == 6, "seeds should be unique per instance"


def test_benchmark_skips_truth_when_disabled():
    inst = sk_instance(8, seed=9, distribution="binary")
    samplers = {"sa": wrap_sa(num_sweeps=50, seed=0)}
    records = benchmark(samplers, [inst], num_reads=5, solve_truth=False)
    r = records[0]
    assert r.ground_state_energy is None
    assert r.success_count == 0
    assert r.tts_99 is None
    assert r.best_energy < float("inf")
