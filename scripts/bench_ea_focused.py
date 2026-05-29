"""Focused 3D EA benchmark at L=5 (N=125) and L=6 (N=216) -- the regime
where PT looked promising in the broader EA sweep.

Uses 10 instances per L for tighter statistical confidence on success rates.
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
    ea_suite,
    records_to_csv,
    records_to_json,
    wrap_dimod,
    wrap_pt,
    wrap_sa,
)

NUM_SWEEPS = 5000
NUM_READS = 50
PT_REPLICAS = 8

SIZES = [5, 6]                  # L values; N = L^3 = 125, 216
INSTANCES_PER_SIZE = 10         # tighter confidence intervals
DIMENSION = 3

REG_PATH = Path("results/sk_registry.json")
CSV_PATH = Path("results/ea_focused_L5_L6.csv")
JSON_PATH = Path("results/ea_focused_L5_L6.json")


def build_samplers():
    return {
        "ising_lab.SA": wrap_sa(num_sweeps=NUM_SWEEPS, seed=0),
        "ising_lab.PT": wrap_pt(
            num_sweeps=NUM_SWEEPS, num_replicas=PT_REPLICAS,
            beta_min=0.1, beta_max=10.0, seed=0,
        ),
        "neal.SA": wrap_dimod(neal.SimulatedAnnealingSampler(), num_sweeps=NUM_SWEEPS),
    }


def rescore_against_panel_best(records, energy_tol=1e-6):
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


def summarize(records, label):
    by_sampler_size: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_sampler_size[r.sampler][r.n].append(r)

    print(f"\n=== {label} ===")
    print(f"{'sampler':<22} {'N':>4} {'p_succ avg':>12} {'p_succ stdev':>14} "
          f"{'TTS_99 (ms)':>14} {'wall/read (ms)':>16} {'best-E gap':>12}")
    print("-" * 100)
    for sampler in sorted(by_sampler_size):
        for n in sorted(by_sampler_size[sampler]):
            rs = by_sampler_size[sampler][n]
            p_vals = [r.success_prob for r in rs]
            p_avg = statistics.mean(p_vals)
            p_std = statistics.stdev(p_vals) if len(p_vals) > 1 else 0.0
            wall_per_read = statistics.mean(
                r.wall_time / max(r.num_reads, 1) for r in rs
            ) * 1000
            tts_finite = [r.tts_99 for r in rs if r.tts_99 is not None]
            tts_str = (
                f"{statistics.median(tts_finite) * 1000:14.2f}"
                if tts_finite else f"{'inf':>14}"
            )
            gaps = [r.best_energy - r.ground_state_energy
                    for r in rs if r.ground_state_energy is not None]
            gap_str = f"{statistics.mean(gaps):+12.2f}" if gaps else f"{'n/a':>12}"
            print(f"{sampler:<22} {n:>4} {p_avg:>12.3f} {p_std:>14.3f} "
                  f"{tts_str} {wall_per_read:>16.2f} {gap_str}")


def main():
    REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    reg = OptimumRegistry(REG_PATH)
    print(f"Loaded {len(reg)} prior records from {REG_PATH}")

    suite = ea_suite(SIZES, instances_per_L=INSTANCES_PER_SIZE,
                     dimension=DIMENSION, base_seed=0)
    samplers = build_samplers()
    print(f"\n3D EA focused: {len(suite)} instances "
          f"(L in {SIZES}, N = {[L**DIMENSION for L in SIZES]}), "
          f"{len(samplers)} samplers, {NUM_READS} reads, {NUM_SWEEPS} sweeps "
          f"({len(suite) * len(samplers) * NUM_READS:,} total reads)")

    records = benchmark(samplers, suite, num_reads=NUM_READS,
                        solve_truth=False, registry=reg)
    rescore_against_panel_best(records)
    summarize(records, f"EA 3D L=5,6 (panel-best as truth, {NUM_SWEEPS} sweeps)")

    csv_out = records_to_csv(records, CSV_PATH)
    json_out = records_to_json(records, JSON_PATH)
    reg_out = reg.save()
    print(f"\nResults: {csv_out} / {json_out}")
    print(f"Registry: {reg_out} ({len(reg)} entries)")

    by_sampler_count: dict = defaultdict(int)
    for inst in suite:
        rec = reg.best(sk_instance_key(inst))
        if rec:
            by_sampler_count[rec.source] += 1
    print(f"\nRegistry floor attribution across {len(suite)} instances:")
    for sampler, count in sorted(by_sampler_count.items(), key=lambda x: -x[1]):
        print(f"  {sampler:<25} {count} instances")


if __name__ == "__main__":
    main()
