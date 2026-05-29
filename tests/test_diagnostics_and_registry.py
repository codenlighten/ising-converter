"""Tests for PT swap-acceptance diagnostics and the OptimumRegistry."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ising_lab import (
    BestKnown,
    IsingModel,
    OptimumRegistry,
    brute_force_ground_state,
    sk_instance_key,
)
from ising_lab.benchmarks import (
    benchmark,
    parallel_tempering_with_diagnostics,
    pt_beta_ladder,
    sk_instance,
    wrap_pt,
    wrap_sa,
)


# ---------- PT diagnostics ----------


def test_pt_beta_ladder_geometric_and_endpoints():
    betas = pt_beta_ladder(0.1, 10.0, 5)
    assert len(betas) == 5
    assert betas[0] == pytest.approx(0.1)
    assert betas[-1] == pytest.approx(10.0)
    # Geometric: ratio between consecutive entries is constant.
    ratios = [betas[k + 1] / betas[k] for k in range(4)]
    assert all(r == pytest.approx(ratios[0]) for r in ratios)


def test_pt_diagnostics_shape_and_sanity():
    """Diagnostics: one dict per read, with sane shapes and value ranges."""
    inst = sk_instance(10, seed=11, distribution="binary")
    diags = parallel_tempering_with_diagnostics(
        inst.model,
        num_sweeps=300,
        num_replicas=6,
        beta_min=0.1,
        beta_max=10.0,
        num_reads=4,
        seed=2026,
    )
    assert len(diags) == 4
    for d in diags:
        assert set(d.keys()) == {
            "state", "energy", "swap_acceptance", "final_energies", "round_trips",
            "n_up", "n_down",
        }
        assert len(d["state"]) == 10
        assert all(s in (-1, 1) for s in d["state"])
        assert len(d["swap_acceptance"]) == 5
        assert all(0.0 <= r <= 1.0 for r in d["swap_acceptance"])
        assert len(d["final_energies"]) == 6
        assert d["energy"] <= min(d["final_energies"]) + 1e-9
        assert inst.model.energy(d["state"]) == pytest.approx(d["energy"])
        # Round trips: one count per replica id, non-negative.
        assert len(d["round_trips"]) == 6
        assert all(t >= 0 for t in d["round_trips"])


def test_pt_round_trips_grow_with_more_sweeps():
    """Longer runs must accumulate more total round trips, all else equal."""
    inst = sk_instance(10, seed=44, distribution="binary")
    short = parallel_tempering_with_diagnostics(
        inst.model, num_sweeps=200, num_replicas=6,
        beta_min=0.1, beta_max=10.0, num_reads=4, seed=1,
    )
    long = parallel_tempering_with_diagnostics(
        inst.model, num_sweeps=4000, num_replicas=6,
        beta_min=0.1, beta_max=10.0, num_reads=4, seed=1,
    )
    short_total = sum(sum(d["round_trips"]) for d in short)
    long_total = sum(sum(d["round_trips"]) for d in long)
    assert long_total > short_total, (
        f"longer run should accumulate more round trips (short={short_total}, long={long_total})"
    )


def test_pt_round_trips_zero_at_tiny_sweep_budget():
    """With only one sweep, no replica can have visited both extremes."""
    inst = sk_instance(8, seed=55, distribution="binary")
    diags = parallel_tempering_with_diagnostics(
        inst.model, num_sweeps=1, num_replicas=6,
        beta_min=0.1, beta_max=10.0, num_reads=2, seed=3,
    )
    for d in diags:
        # A label change requires touching one extreme, then the other.
        # One sweep + one swap step is not enough to round-trip.
        assert sum(d["round_trips"]) == 0


def test_pt_diagnostics_cold_replica_has_lowest_average_final_energy():
    """At equilibrium, the cold replica's final energy is on average lowest.

    A weak but robust physical sanity check: averaged across reads, the final
    energy at beta_max should be at or below the final energy at beta_min.
    Useful as a smoke test that the swap mechanism isn't accidentally inverted.
    """
    inst = sk_instance(12, seed=33, distribution="binary")
    diags = parallel_tempering_with_diagnostics(
        inst.model, num_sweeps=800, num_replicas=8,
        beta_min=0.1, beta_max=10.0, num_reads=16, seed=99,
    )
    hot = sum(d["final_energies"][0] for d in diags) / len(diags)
    cold = sum(d["final_energies"][-1] for d in diags) / len(diags)
    assert cold <= hot, (
        f"cold-replica avg final energy ({cold:.2f}) should be <= hot ({hot:.2f})"
    )


# ---------- OptimumRegistry ----------


def test_registry_update_is_monotone_in_energy():
    reg = OptimumRegistry()
    state = [1, -1, 1, -1]
    assert reg.update("k", -1.0, state, "sa") is True
    # Worse-energy update is rejected.
    assert reg.update("k", -0.5, state, "pt") is False
    assert reg.best("k").energy == pytest.approx(-1.0)
    assert reg.best("k").source == "sa"
    # Strictly better wins.
    assert reg.update("k", -2.0, state, "pt") is True
    assert reg.best("k").source == "pt"
    assert reg.best("k").energy == pytest.approx(-2.0)


def test_registry_tie_does_not_replace():
    """Ties (within tol) should leave the original record in place."""
    reg = OptimumRegistry(tol=1e-9)
    reg.update("k", -3.0, [1, 1], "sa")
    # An equal-energy update from another sampler must not displace the first.
    assert reg.update("k", -3.0, [-1, -1], "pt") is False
    assert reg.best("k").source == "sa"


def test_registry_persistence_round_trip(tmp_path: Path):
    """save -> load returns equivalent records."""
    p = tmp_path / "reg.json"
    reg = OptimumRegistry(p)
    reg.update("a", -1.5, [1, -1, 1], "sa", metadata={"n": 3})
    reg.update("b", -2.0, [1, 1], "pt", metadata={"n": 2})
    reg.save()
    # The file should be human-readable JSON with the expected keys.
    payload = json.loads(p.read_text())
    assert set(payload.keys()) == {"a", "b"}

    reg2 = OptimumRegistry(p)
    assert len(reg2) == 2
    assert reg2.best("a").energy == pytest.approx(-1.5)
    assert reg2.best("a").source == "sa"
    assert reg2.best("a").state == [1, -1, 1]
    assert reg2.best("a").metadata == {"n": 3}
    assert reg2.best("b").energy == pytest.approx(-2.0)


def test_registry_seeds_truth_in_benchmark(tmp_path: Path):
    """benchmark() should use the registry's energy as truth, no brute force needed."""
    inst = sk_instance(10, seed=77, distribution="binary")
    # Pre-populate registry with the exact ground state.
    truth_state, truth_energy = brute_force_ground_state(inst.model)
    reg = OptimumRegistry(tmp_path / "r.json")
    reg.update(
        sk_instance_key(inst),
        energy=truth_energy,
        state=truth_state,
        source="brute_force",
    )
    # Run benchmark with solve_truth=False -- only the registry can supply truth.
    records = benchmark(
        {"pt": wrap_pt(num_sweeps=400, num_replicas=6, seed=3)},
        [inst],
        num_reads=20,
        solve_truth=False,
        registry=reg,
    )
    r = records[0]
    assert r.ground_state_energy == pytest.approx(truth_energy)
    # PT should find the registry-stored truth at least once on N=10.
    assert r.success_count >= 1


def test_benchmark_updates_registry_when_sampler_beats_existing(tmp_path: Path):
    """If a sampler finds a strictly lower energy than the registry, source flips."""
    inst = sk_instance(10, seed=88, distribution="binary")
    truth_state, truth_energy = brute_force_ground_state(inst.model)
    reg = OptimumRegistry(tmp_path / "r.json")
    # Seed registry with a deliberately too-high energy attributed to a fake source.
    reg.update(
        sk_instance_key(inst),
        energy=truth_energy + 5.0,  # worse than what PT will find
        state=truth_state,
        source="fake-old-result",
    )
    records = benchmark(
        {"pt": wrap_pt(num_sweeps=400, num_replicas=6, seed=4)},
        [inst],
        num_reads=20,
        solve_truth=False,
        registry=reg,
    )
    new_best = reg.best(sk_instance_key(inst))
    assert new_best.source == "pt"
    assert new_best.energy <= truth_energy + 1e-9
    # The benchmark record's success_count is computed against the stale truth
    # that was seeded at the start of the call, so "improvement" here means
    # registry source flipped, not that the record's metric changed.
    assert records[0].sampler == "pt"
