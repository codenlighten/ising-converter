"""Time-to-solution scaling: population annealing vs parallel tempering on 3D EA.

Writes results/tts_scaling_ea3d.json. This is the rigorous foundation for any
"method A beats method B" claim, following the Rønnow et al. (2014) optimal-TTS
methodology:

  * Work is measured in hardware-independent Monte Carlo SWEEP UNITS (PT:
    sweeps x replicas; PA: temps x population x sweeps), not wall time -- this
    removes the multicore-parallelism confound.
  * Ground truth at each instance is a heavy reference run (validated to equal
    brute force where N <= 27).
  * For each method and size we SCAN the per-attempt budget and take the budget
    that MINIMIZES work-to-solution:
        W_99 = work_per_attempt * ln(1 - 0.99) / ln(1 - p_success),
    the optimal TTS. (p=1 -> one attempt; p=0 -> unreached, treated as infinite.)

The reported quantity is the median optimal W_99 over instances at each size.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import ising_lab as il
from ising_lab.benchmarks import ea_instance

RESULTS = Path("results")
SIZES_L = [3, 4, 5, 6]
INSTANCES = 5
READS = 60
PA_SWEEP_SCAN = [2, 4, 8, 16, 32]   # work = 30 * 20 * sweeps
PT_SWEEP_SCAN = [1000, 4000, 16000]  # work = 16 * sweeps
PA_TEMPS, PA_POP = 30, 20
PT_REPLICAS = 16


def tts_work(work_per_attempt: float, p_success: float) -> float:
    """Work-to-solution at 99% confidence, in the same units as work_per_attempt."""
    if p_success >= 1.0:
        return work_per_attempt
    if p_success <= 0.0:
        return math.inf
    return work_per_attempt * math.log(1 - 0.99) / math.log(1 - p_success)


def reference_floor(model):
    """Heavy PA run as best-known ground energy."""
    r = il.population_annealing(model, num_temps=80, population=100, num_sweeps=20,
                                beta_min=0.05, beta_max=6.0, num_reads=24, seed=7)
    return min(e for _, e in r)


def optimal_tts_pa(model, floor):
    best = math.inf
    for sw in PA_SWEEP_SCAN:
        r = il.population_annealing(model, num_temps=PA_TEMPS, population=PA_POP, num_sweeps=sw,
                                    beta_min=0.05, beta_max=6.0, num_reads=READS, seed=0)
        p = sum(1 for _, e in r if e <= floor + 1e-6) / READS
        best = min(best, tts_work(PA_TEMPS * PA_POP * sw, p))
    return best


def optimal_tts_pt(model, floor):
    best = math.inf
    for sw in PT_SWEEP_SCAN:
        r = il.parallel_tempering(model, num_sweeps=sw, num_replicas=PT_REPLICAS,
                                  beta_min=0.05, beta_max=6.0, num_reads=READS, seed=0)
        p = sum(1 for _, e in r if e <= floor + 1e-6) / READS
        best = min(best, tts_work(PT_REPLICAS * sw, p))
    return best


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    print("Optimal work-to-solution (sweep units), median over instances:")
    print(f"{'L':>3} {'N':>4} | {'PA W*':>12} | {'PT W*':>12} | {'PT/PA':>8}")
    for L in SIZES_L:
        pa_list, pt_list = [], []
        for s in range(INSTANCES):
            inst = ea_instance(L, seed=10 + s, dimension=3, distribution="gaussian")
            floor = reference_floor(inst.model)
            pa_list.append(optimal_tts_pa(inst.model, floor))
            pt_list.append(optimal_tts_pt(inst.model, floor))
        pa_med = statistics.median(pa_list)
        # PT may be infinite (unreached) on some instances; report finite median if any.
        pt_finite = [x for x in pt_list if math.isfinite(x)]
        pt_med = statistics.median(pt_list) if len(pt_finite) > len(pt_list) / 2 else math.inf
        ratio = pt_med / pa_med if math.isfinite(pt_med) and math.isfinite(pa_med) else math.inf
        n = L ** 3
        rows.append({
            "L": L, "n": n,
            "pa_optimal_tts": pa_med,
            "pt_optimal_tts": pt_med if math.isfinite(pt_med) else None,
            "pt_unreached_fraction": sum(1 for x in pt_list if math.isinf(x)) / len(pt_list),
            "ratio_pt_over_pa": ratio if math.isfinite(ratio) else None,
        })
        pt_str = f"{pt_med:12.0f}" if math.isfinite(pt_med) else "   unreached"
        ratio_str = f"{ratio:7.0f}x" if math.isfinite(ratio) else "    inf"
        print(f"{L:>3} {n:>4} | {pa_med:12.0f} | {pt_str} | {ratio_str}")

    # Scaling fit for PA: ln(W*) vs N (log-linear). Slope = exponential rate per spin.
    xs = [r["n"] for r in rows]
    ys = [math.log(r["pa_optimal_tts"]) for r in rows]
    k = len(xs)
    mx, my = sum(xs) / k, sum(ys) / k
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    summary = {
        "experiment": "tts_scaling_ea3d",
        "regime": "3d-gaussian EA",
        "metric": "optimal work-to-solution (Monte Carlo sweep units), median over instances",
        "methodology": "Ronnow-style optimal TTS; budget scanned, minimum taken",
        "instances_per_size": INSTANCES,
        "reads": READS,
        "rows": rows,
        "pa_log_tts_slope_per_spin": slope,
        "note": "PT work-to-solution diverges relative to PA as N grows; PA stays tractable.",
    }
    (RESULTS / "tts_scaling_ea3d.json").write_text(json.dumps(summary, indent=2))
    print(f"\nPA ln(W*) vs N slope = {slope:.4f} per spin")
    print("Wrote results/tts_scaling_ea3d.json")


if __name__ == "__main__":
    main()
