"""PA vs PT-ICM on 2D degree-5 high-precision spin glasses (D-Wave proxy class).

Writes results/pa_vs_pticm_degree5_2d.json. The instance class is a 5-regular
2D toroidal lattice with Gaussian couplings -- a matched-in-spirit proxy for the
2D degree-5 high-precision graph on which D-Wave's strongest optimization
scaling-advantage claim is benchmarked (Munoz Bauza & Lidar, PRL 134, 160601,
2025). It is NOT D-Wave's exact QAC graph.

Metric: optimal work-to-solution in Monte Carlo sweep units (budget scanned,
minimum taken), median over instances. PT-ICM is given a large extra budget to
be fair. Ground truth is a heavy-PA floor.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import ising_lab as il
from ising_lab.benchmarks import degree5_2d_instance

RESULTS = Path("results")
SIZES_L = [8, 10, 12, 14]
INSTANCES = 4
READS = 60
PA_SWEEP_SCAN = [4, 8, 16, 32]          # work = 30 * 20 * sweeps
PTICM_SWEEP_SCAN = [4000, 16000, 64000]  # work = 2 lanes * 16 replicas * sweeps


def tts_work(work, p):
    if p >= 1.0:
        return work
    if p <= 0.0:
        return math.inf
    return work * math.log(0.01) / math.log(1 - p)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    print("PA vs PT-ICM on 2D degree-5 Gaussian (proxy for the D-Wave class)")
    print(f"{'L':>3} {'N':>4} | {'PA W*':>10} | {'PT-ICM W*':>12} | ratio")
    for L in SIZES_L:
        pa_l, icm_l = [], []
        for s in range(INSTANCES):
            inst = degree5_2d_instance(L, seed=10 + s, distribution="gaussian")
            floor = min(e for _, e in il.population_annealing(
                inst.model, num_temps=80, population=100, num_sweeps=20,
                beta_min=0.05, beta_max=6.0, num_reads=24, seed=7))
            pa_best = math.inf
            for sw in PA_SWEEP_SCAN:
                r = il.population_annealing(inst.model, num_temps=30, population=20, num_sweeps=sw,
                                            beta_min=0.05, beta_max=6.0, num_reads=READS, seed=0)
                p = sum(1 for _, e in r if e <= floor + 1e-6) / READS
                pa_best = min(pa_best, tts_work(30 * 20 * sw, p))
            icm_best = math.inf
            for sw in PTICM_SWEEP_SCAN:
                r = il.parallel_tempering_houdayer(inst.model, num_sweeps=sw, num_replicas=16,
                                                   beta_min=0.05, beta_max=6.0, icm_every=5,
                                                   num_reads=READS, seed=0)
                p = sum(1 for _, e in r if e <= floor + 1e-6) / READS
                icm_best = min(icm_best, tts_work(2 * 16 * sw, p))
            pa_l.append(pa_best)
            icm_l.append(icm_best)
        pa_med = statistics.median(pa_l)
        icm_fin = [x for x in icm_l if math.isfinite(x)]
        icm_med = statistics.median(icm_l) if len(icm_fin) > len(icm_l) / 2 else math.inf
        ratio = icm_med / pa_med if math.isfinite(icm_med) and math.isfinite(pa_med) else math.inf
        rows.append({
            "L": L, "n": L * L,
            "pa_optimal_tts": pa_med,
            "pticm_optimal_tts": icm_med if math.isfinite(icm_med) else None,
            "pticm_unreached_fraction": sum(1 for x in icm_l if math.isinf(x)) / len(icm_l),
            "ratio_pticm_over_pa": ratio if math.isfinite(ratio) else None,
        })
        icm_str = f"{icm_med:12.0f}" if math.isfinite(icm_med) else "   unreached"
        ratio_str = f"{ratio:7.0f}x" if math.isfinite(ratio) else "    inf"
        print(f"{L:>3} {L*L:>4} | {pa_med:10.0f} | {icm_str} | {ratio_str}")

    summary = {
        "experiment": "pa_vs_pticm_degree5_2d",
        "instance_class": "5-regular 2D toroidal lattice, Gaussian couplings (proxy for D-Wave 2D degree-5 high-precision; NOT the exact QAC graph)",
        "metric": "optimal work-to-solution (MC sweep units), median over instances",
        "ground_truth": "heavy-PA floor",
        "instances_per_size": INSTANCES,
        "reads": READS,
        "rows": rows,
        "caveat": "PT-ICM uses fixed hyperparameters (16 replicas, beta 0.05-6.0) scanned over sweeps; a per-instance-tuned PT-ICM (as in the D-Wave benchmark) could do better. The PA-vs-PT-ICM comparison is apples-to-apples within this fixed-hyperparameter framework.",
    }
    (RESULTS / "pa_vs_pticm_degree5_2d.json").write_text(json.dumps(summary, indent=2))
    print("\nWrote results/pa_vs_pticm_degree5_2d.json")


if __name__ == "__main__":
    main()
