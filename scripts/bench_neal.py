"""Run an apples-to-apples benchmark of ising_lab's SA + PT against neal's SA
on a binary Sherrington-Kirkpatrick suite.

The three samplers all consume the same dimod-style BQM via the registry/wrap
plumbing, so reported energies are directly comparable. Wall times are wall
clock as observed by the harness (per-read time = wall_time / num_reads).
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path

import neal

from ising_lab import OptimumRegistry, sk_instance_key
from ising_lab.benchmarks import (
    benchmark,
    records_to_csv,
    records_to_json,
    sk_suite,
    wrap_dimod,
    wrap_pt,
    wrap_sa,
)

NUM_SWEEPS_SMALL = 1000     # matched between all SA-style samplers (small N)
NUM_SWEEPS_LARGE = 5000     # at N >= 30 the landscape gets harder; budget more
PT_REPLICAS = 8             # PT has R replicas, so it's ~R times more compute per chain
NUM_READS = 50

# Brute-forceable sizes get a definitive ground truth in the registry.
SMALL_SIZES = [12, 16, 20, 24]
# Pushed beyond brute force: panel-best is the floor.
LARGE_SIZES = [30, 50, 80, 100, 150, 200, 300]

INSTANCES_PER_SIZE = 5
REG_PATH = Path("results/sk_registry.json")
CSV_PATH = Path("results/sk_neal_vs_ising_lab.csv")
JSON_PATH = Path("results/sk_neal_vs_ising_lab.json")


def build_samplers(num_sweeps: int):
    return {
        "ising_lab.SA": wrap_sa(num_sweeps=num_sweeps, seed=0),
        "ising_lab.PT": wrap_pt(num_sweeps=num_sweeps, num_replicas=PT_REPLICAS, seed=0),
        "neal.SA":      wrap_dimod(neal.SimulatedAnnealingSampler(), num_sweeps=num_sweeps),
    }


def rescore_against_panel_best(records, energy_tol=1e-6):
    """Recompute each record's success metrics using min(best_energy) across
    all samplers on the same instance as the truth.

    This makes the report fair when no brute-force truth is available -- the
    first sampler doesn't get penalized for running before any floor exists.
    Mutates records in place and returns them.
    """
    panel_best: dict = defaultdict(lambda: float("inf"))
    for r in records:
        key = (r.instance_seed, r.n)
        panel_best[key] = min(panel_best[key], r.best_energy)

    for r in records:
        truth = panel_best[(r.instance_seed, r.n)]
        r.ground_state_energy = truth
        r.success_count = sum(1 for e in r.energies if e <= truth + energy_tol)
        r.success_prob = r.success_count / r.num_reads if r.num_reads > 0 else 0.0
        if r.num_reads == 0:
            r.tts_99 = None
        elif r.success_prob >= 1.0:
            r.tts_99 = r.wall_time / r.num_reads
        elif r.success_prob <= 0.0:
            r.tts_99 = None
        else:
            r.tts_99 = (r.wall_time / r.num_reads) * math.log(0.01) / math.log(1.0 - r.success_prob)
    return records


def summarize(records, label):
    by_sampler_size: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_sampler_size[r.sampler][r.n].append(r)

    print(f"\n=== {label} ===")
    print(f"{'sampler':<15} {'N':>4} {'p_succ (avg)':>14} {'TTS_99 (ms)':>14} "
          f"{'wall/read (ms)':>16} {'best-E gap':>12}")
    print("-" * 80)
    for sampler in sorted(by_sampler_size):
        for n in sorted(by_sampler_size[sampler]):
            rs = by_sampler_size[sampler][n]
            p_avg = statistics.mean(r.success_prob for r in rs)
            wall_per_read = statistics.mean(
                r.wall_time / max(r.num_reads, 1) for r in rs
            ) * 1000
            tts_finite = [r.tts_99 for r in rs if r.tts_99 is not None]
            tts_str = (
                f"{statistics.median(tts_finite) * 1000:14.2f}"
                if tts_finite else f"{'inf':>14}"
            )
            # Gap from registry's best (negative or zero == matched the floor).
            gaps = []
            for r in rs:
                if r.ground_state_energy is not None:
                    gaps.append(r.best_energy - r.ground_state_energy)
            gap_str = (
                f"{statistics.mean(gaps):+12.2f}" if gaps else f"{'n/a':>12}"
            )
            print(f"{sampler:<15} {n:>4} {p_avg:>14.3f} {tts_str} "
                  f"{wall_per_read:>16.2f} {gap_str}")


def main():
    REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    reg = OptimumRegistry(REG_PATH)
    print(f"Loaded {len(reg)} prior records from {REG_PATH}")

    # Pass 1: small (brute-forceable) instances, light sweep budget.
    small = sk_suite(SMALL_SIZES, instances_per_size=INSTANCES_PER_SIZE, base_seed=0)
    samplers_small = build_samplers(NUM_SWEEPS_SMALL)
    print(f"\nRunning small suite: {len(small)} instances, "
          f"{len(samplers_small)} samplers, {NUM_READS} reads each, "
          f"{NUM_SWEEPS_SMALL} sweeps "
          f"({len(small) * len(samplers_small) * NUM_READS:,} total reads)")
    small_records = benchmark(samplers_small, small, num_reads=NUM_READS, registry=reg)
    summarize(small_records, f"Small instances (brute-force truth, {NUM_SWEEPS_SMALL} sweeps)")

    # Pass 2: large instances, heavier sweep budget. No brute force.
    large = sk_suite(LARGE_SIZES, instances_per_size=INSTANCES_PER_SIZE, base_seed=0)
    samplers_large = build_samplers(NUM_SWEEPS_LARGE)
    print(f"\nRunning large suite: {len(large)} instances, "
          f"{len(samplers_large)} samplers, {NUM_READS} reads each, "
          f"{NUM_SWEEPS_LARGE} sweeps")
    large_records = benchmark(samplers_large, large, num_reads=NUM_READS,
                              solve_truth=False, registry=reg)
    rescore_against_panel_best(large_records)
    summarize(large_records, f"Large instances (panel-best as truth, {NUM_SWEEPS_LARGE} sweeps)")

    # Persist
    all_records = small_records + large_records
    csv_out = records_to_csv(all_records, CSV_PATH)
    json_out = records_to_json(all_records, JSON_PATH)
    reg_out = reg.save()

    print(f"\nResults:")
    print(f"  CSV:  {csv_out} ({csv_out.stat().st_size} bytes)")
    print(f"  JSON: {json_out} ({json_out.stat().st_size} bytes)")
    print(f"  Registry: {reg_out} ({reg_out.stat().st_size} bytes, {len(reg)} entries)")

    # Show which sampler currently holds the floor for each large instance.
    print(f"\nRegistry floor for large instances ({LARGE_SIZES}):")
    for inst in large:
        key = sk_instance_key(inst)
        rec = reg.best(key)
        if rec:
            print(f"  {key:<28}  E={rec.energy:+8.1f}  source={rec.source}")


if __name__ == "__main__":
    main()
