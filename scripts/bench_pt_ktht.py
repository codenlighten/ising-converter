"""KTHT-tuned PT vs default PT vs SA at N=200, 300.

Uses the Katzgraber-Trebst-Huse-Troyer feedback-optimized beta ladder
auto-tuner from ising_lab.benchmarks.auto_tune_beta_ladder_ktht. Pilot cost
is included in the sampler's wall time for fair TTS comparison.
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
    wrap_pt_ktht,
    wrap_sa,
)

NUM_SWEEPS = 5000
NUM_READS = 50
PT_REPLICAS = 8

SIZES = [200, 300]
INSTANCES_PER_SIZE = 5

REG_PATH = Path("results/sk_registry.json")
CSV_PATH = Path("results/pt_ktht_vs_baseline.csv")
JSON_PATH = Path("results/pt_ktht_vs_baseline.json")


def build_samplers():
    return {
        "ising_lab.SA": wrap_sa(num_sweeps=NUM_SWEEPS, seed=0),
        "ising_lab.PT": wrap_pt(
            num_sweeps=NUM_SWEEPS, num_replicas=PT_REPLICAS,
            beta_min=0.1, beta_max=10.0, seed=0,
        ),
        "ising_lab.PT-ktht": wrap_pt_ktht(
            num_sweeps=NUM_SWEEPS, num_replicas=PT_REPLICAS,
            beta_min=0.1, beta_max=10.0,
            pilot_sweeps=1000, pilot_reads=4, num_iterations=5,
            seed=0,
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
    return records


def summarize(records, label):
    by_sampler_size: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_sampler_size[r.sampler][r.n].append(r)

    print(f"\n=== {label} ===")
    print(f"{'sampler':<22} {'N':>4} {'p_succ (avg)':>14} {'TTS_99 (ms)':>14} "
          f"{'wall/read (ms)':>16} {'best-E gap':>12}")
    print("-" * 86)
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
            gaps = [r.best_energy - r.ground_state_energy
                    for r in rs if r.ground_state_energy is not None]
            gap_str = f"{statistics.mean(gaps):+12.2f}" if gaps else f"{'n/a':>12}"
            print(f"{sampler:<22} {n:>4} {p_avg:>14.3f} {tts_str} "
                  f"{wall_per_read:>16.2f} {gap_str}")


def main():
    REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    reg = OptimumRegistry(REG_PATH)
    print(f"Loaded {len(reg)} prior records from {REG_PATH}")

    suite = sk_suite(SIZES, instances_per_size=INSTANCES_PER_SIZE, base_seed=0)
    samplers = build_samplers()
    print(f"\nRunning suite: {len(suite)} instances, {len(samplers)} samplers, "
          f"{NUM_READS} reads each, {NUM_SWEEPS} sweeps "
          f"({len(suite) * len(samplers) * NUM_READS:,} total reads)")

    records = benchmark(samplers, suite, num_reads=NUM_READS,
                        solve_truth=False, registry=reg)
    rescore_against_panel_best(records)
    summarize(records, f"KTHT vs default (panel-best as truth, {NUM_SWEEPS} sweeps)")

    csv_out = records_to_csv(records, CSV_PATH)
    json_out = records_to_json(records, JSON_PATH)
    reg_out = reg.save()

    print(f"\nResults:")
    print(f"  CSV:  {csv_out}")
    print(f"  JSON: {json_out}")
    print(f"  Registry: {reg_out} ({len(reg)} entries)")

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
