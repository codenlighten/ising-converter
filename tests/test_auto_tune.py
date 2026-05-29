"""Tests for parallel_tempering_betas + auto_tune_beta_ladder."""
from __future__ import annotations

import statistics

import pytest

from ising_lab.benchmarks import (
    auto_tune_beta_ladder,
    parallel_tempering_betas,
    pt_beta_ladder,
    sk_instance,
)


def test_parallel_tempering_betas_accepts_custom_ladder():
    """A non-geometric ladder runs and returns the right diagnostic shape."""
    inst = sk_instance(10, seed=1, distribution="binary")
    custom = [0.1, 0.3, 1.0, 5.0, 10.0]  # not geometric
    diags = parallel_tempering_betas(
        inst.model, betas=custom, num_sweeps=200, num_reads=3, seed=42,
    )
    assert len(diags) == 3
    for d in diags:
        assert len(d["state"]) == 10
        assert len(d["swap_acceptance"]) == len(custom) - 1
        assert len(d["final_energies"]) == len(custom)
        assert len(d["round_trips"]) == len(custom)


def test_parallel_tempering_betas_rejects_invalid_ladder():
    inst = sk_instance(6, seed=2, distribution="binary")
    with pytest.raises(Exception):
        parallel_tempering_betas(inst.model, betas=[1.0], num_sweeps=10)
    with pytest.raises(Exception):
        parallel_tempering_betas(inst.model, betas=[1.0, 0.5], num_sweeps=10)  # not increasing
    with pytest.raises(Exception):
        parallel_tempering_betas(inst.model, betas=[-1.0, 1.0], num_sweeps=10)  # non-positive


def test_auto_tune_preserves_endpoints_and_monotonicity():
    inst = sk_instance(12, seed=3, distribution="binary")
    betas, history = auto_tune_beta_ladder(
        inst.model,
        num_replicas=6,
        beta_min=0.1,
        beta_max=10.0,
        target_acceptance=0.3,
        pilot_sweeps=200,
        pilot_reads=4,
        num_iterations=5,
        seed=7,
    )
    assert len(betas) == 6
    assert betas[0] == pytest.approx(0.1)
    assert betas[-1] == pytest.approx(10.0)
    # Strictly increasing.
    for k in range(5):
        assert betas[k] < betas[k + 1]
    # History has at most num_iterations entries.
    assert 1 <= len(history) <= 5
    assert all(len(row) == 5 for row in history)


def test_auto_tune_produces_nontrivial_ladder():
    """The tuned ladder should differ meaningfully from the geometric starting
    point when the geometric ladder is visibly unbalanced."""
    inst = sk_instance(20, seed=4, distribution="binary")
    R = 8

    geometric = pt_beta_ladder(0.1, 10.0, R)
    tuned, history = auto_tune_beta_ladder(
        inst.model,
        num_replicas=R,
        beta_min=0.1,
        beta_max=10.0,
        target_acceptance=0.3,
        pilot_sweeps=500,
        pilot_reads=6,
        num_iterations=5,
        seed=10,
    )
    # Interior points must have moved (this problem has a clearly unbalanced
    # geometric ladder, so the tuner should shift things).
    max_shift = max(abs(g - t) for g, t in zip(geometric, tuned))
    assert max_shift > 0.05, f"tuner did not move interior points (max shift {max_shift:.3f})"
    # The tuner ran at least once.
    assert len(history) >= 1


def test_auto_tune_does_not_worsen_bottleneck():
    """The slowest pair's swap acceptance should not get drastically worse.
    A correct tuner can leave some pairs imbalanced (especially saturated ones),
    but it should not turn an already-bad pair into a much worse one."""
    inst = sk_instance(20, seed=4, distribution="binary")
    R = 8

    geometric = pt_beta_ladder(0.1, 10.0, R)
    baseline = parallel_tempering_betas(
        inst.model, betas=geometric, num_sweeps=1000, num_reads=8, seed=100,
    )
    base_avg = [
        statistics.mean(d["swap_acceptance"][k] for d in baseline)
        for k in range(R - 1)
    ]

    tuned, _ = auto_tune_beta_ladder(
        inst.model,
        num_replicas=R,
        beta_min=0.1,
        beta_max=10.0,
        target_acceptance=0.3,
        pilot_sweeps=500,
        pilot_reads=6,
        num_iterations=5,
        seed=200,
    )
    after = parallel_tempering_betas(
        inst.model, betas=tuned, num_sweeps=1000, num_reads=8, seed=300,
    )
    tuned_avg = [
        statistics.mean(d["swap_acceptance"][k] for d in after)
        for k in range(R - 1)
    ]

    assert min(tuned_avg) >= min(base_avg) - 0.15, (
        f"tuned bottleneck {min(tuned_avg):.3f} regressed too far from "
        f"baseline {min(base_avg):.3f}; baseline={base_avg}, tuned={tuned_avg}"
    )


def test_auto_tune_converges_quickly_when_already_balanced():
    """If the initial geometric ladder happens to already be within tolerance,
    auto_tune should stop early -- history length tells us when."""
    inst = sk_instance(10, seed=5, distribution="binary")
    # Very loose tolerance: virtually any rate within [0, 0.5+] passes.
    betas, history = auto_tune_beta_ladder(
        inst.model,
        num_replicas=4,
        beta_min=0.5,
        beta_max=2.0,
        target_acceptance=0.5,
        pilot_sweeps=200,
        pilot_reads=4,
        num_iterations=10,
        tolerance=0.5,  # essentially "any acceptance passes"
        seed=11,
    )
    # Should converge on iteration 1.
    assert len(history) == 1
