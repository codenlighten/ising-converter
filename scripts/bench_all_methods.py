"""Unified comparison of every method in the lab, across three regimes.

Writes results/all_methods_comparison.json. The samplers are SA, PT,
Houdayer-PT, and population annealing; belief propagation is reported as a
deterministic baseline (its rounded-marginal ground-state estimate), not as a
sampler -- it does not fit the (model, num_reads) contract and is weak at
optimization on frustrated graphs.

Three regimes:
  1. SK binary, brute-forceable N (16, 24): exact truth, so we report
     success probability and TTS_99 via the benchmark harness.
  2. 3D EA Gaussian (L=6, N=216): PA's home turf -- best and mean energy,
     wall time, plus the BP baseline.
  3. SK binary N=200: beyond brute force -- best energy density vs the Parisi
     thermodynamic-limit constant -0.7632.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import ising_lab as il
from ising_lab.benchmarks import (
    PARISI_SK_ENERGY_DENSITY,
    benchmark,
    ea_instance,
    sk_energy_density,
    sk_instance,
    sk_suite,
    wrap_pa,
    wrap_pt,
    wrap_pt_houdayer,
    wrap_sa,
)
from ising_lab.inference import bp_ground_state
from ising_lab.registry import OptimumRegistry

RESULTS = Path("results")


def regime_sk_bruteforce() -> dict:
    """SK binary at brute-forceable N: success prob + TTS against exact truth."""
    samplers = {
        "SA": wrap_sa(num_sweeps=2000, seed=0),
        "PT": wrap_pt(num_sweeps=2000, num_replicas=16, beta_min=0.05, beta_max=8.0, seed=0),
        "Houdayer-PT": wrap_pt_houdayer(num_sweeps=2000, num_replicas=16, beta_min=0.05,
                                        beta_max=8.0, icm_every=10, seed=0),
        "PA": wrap_pa(num_temps=40, population=30, num_sweeps=10, beta_min=0.05, beta_max=8.0, seed=0),
    }
    instances = sk_suite([16, 24], instances_per_size=5, distribution="binary")
    reg = OptimumRegistry()
    records = benchmark(samplers, instances, num_reads=30, registry=reg)
    out = {}
    for name in samplers:
        for n in (16, 24):
            rs = [r for r in records if r.sampler == name and r.n == n]
            out.setdefault(name, {})[f"N{n}"] = {
                "success_prob": statistics.mean(r.success_prob for r in rs),
                "median_tts_99": statistics.median(
                    [r.tts_99 for r in rs if r.tts_99 is not None] or [float("nan")]
                ),
                "wall_time": statistics.mean(r.wall_time for r in rs),
            }
    print("Regime 1 -- SK binary, exact truth (success prob | median TTS_99 s):")
    for name in samplers:
        cells = " | ".join(
            f"N{n}: {out[name][f'N{n}']['success_prob']*100:3.0f}%  {out[name][f'N{n}']['median_tts_99']:.4f}s"
            for n in (16, 24)
        )
        print(f"  {name:>12}: {cells}")
    return out


def regime_ea_gaussian() -> dict:
    """3D EA Gaussian L=6: best/mean energy + wall time, plus BP baseline."""
    L, READS = 6, 16
    methods = {
        "SA": lambda m: il.simulated_anneal(m, num_sweeps=4000, num_reads=READS, beta_start=0.05, beta_end=6.0, seed=0),
        "PT": lambda m: il.parallel_tempering(m, num_sweeps=2000, num_replicas=24, beta_min=0.05, beta_max=6.0, num_reads=READS, seed=0),
        "Houdayer-PT": lambda m: il.parallel_tempering_houdayer(m, num_sweeps=2000, num_replicas=24, beta_min=0.05, beta_max=6.0, icm_every=5, num_reads=READS, seed=0),
        "PA": lambda m: il.population_annealing(m, num_temps=50, population=40, num_sweeps=8, beta_min=0.05, beta_max=6.0, num_reads=READS, seed=0),
    }
    insts = [ea_instance(L, seed=700 + s, dimension=3, distribution="gaussian") for s in range(5)]
    out = {}
    for name, fn in methods.items():
        bests, means, times = [], [], []
        for inst in insts:
            t = time.perf_counter()
            r = fn(inst.model)
            times.append(time.perf_counter() - t)
            e = [x for _, x in r]
            bests.append(min(e))
            means.append(statistics.mean(e))
        out[name] = {"mean_best": statistics.mean(bests), "mean_mean": statistics.mean(means),
                     "mean_wall_time": statistics.mean(times)}
    # BP deterministic baseline.
    bp_bests, bp_times = [], []
    for inst in insts:
        t = time.perf_counter()
        best = min(bp_ground_state(inst.model, beta=b)[1] for b in (1.0, 2.0, 4.0))
        bp_times.append(time.perf_counter() - t)
        bp_bests.append(best)
    out["BP (baseline)"] = {"mean_best": statistics.mean(bp_bests), "mean_mean": float("nan"),
                            "mean_wall_time": statistics.mean(bp_times)}
    print(f"\nRegime 2 -- 3D EA Gaussian L={L} (N={L**3}); mean best / mean typical / wall:")
    for name, d in out.items():
        print(f"  {name:>14}: best={d['mean_best']:8.1f}  typ={d['mean_mean']:8.1f}  t={d['mean_wall_time']:.2f}s")
    return out


def regime_sk_parisi() -> dict:
    """SK binary N=200: best energy density vs Parisi -0.7632."""
    n = 200
    methods = {
        "SA": lambda m: il.simulated_anneal(m, num_sweeps=8000, num_reads=16, beta_start=0.05, beta_end=8.0, seed=0),
        "PT": lambda m: il.parallel_tempering(m, num_sweeps=4000, num_replicas=24, beta_min=0.05, beta_max=8.0, num_reads=16, seed=0),
        "PA": lambda m: il.population_annealing(m, num_temps=60, population=40, num_sweeps=8, beta_min=0.05, beta_max=8.0, num_reads=16, seed=0),
    }
    insts = [sk_instance(n, seed=800 + s, distribution="binary") for s in range(3)]
    out = {}
    for name, fn in methods.items():
        dens = []
        for inst in insts:
            r = fn(inst.model)
            dens.append(sk_energy_density(min(x for _, x in r), n))
        out[name] = {"mean_energy_density": statistics.mean(dens),
                     "gap_to_parisi": statistics.mean(dens) + PARISI_SK_ENERGY_DENSITY}
    print(f"\nRegime 3 -- SK binary N={n}; energy density (Parisi = -{PARISI_SK_ENERGY_DENSITY}):")
    for name, d in out.items():
        print(f"  {name:>12}: e={d['mean_energy_density']:.4f}  gap={d['gap_to_parisi']:+.4f}")
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    summary = {
        "sk_bruteforce": regime_sk_bruteforce(),
        "ea_gaussian_L6": regime_ea_gaussian(),
        "sk_parisi_N200": regime_sk_parisi(),
    }
    (RESULTS / "all_methods_comparison.json").write_text(json.dumps(summary, indent=2))
    print("\nWrote results/all_methods_comparison.json")


if __name__ == "__main__":
    main()
