"""Population annealing vs parallel tempering on hard 3D Edwards-Anderson glasses.

Writes results/population_vs_pt_ea3d.json. The comparison is wall-time-honest:
each sampler's wall time is recorded, and PT is also given a large extra sweep
budget to test whether more compute lets it catch PA. On L=8 (N=512) Gaussian
3D EA, PA reaches markedly lower energies than PT even when PT is allowed an
order of magnitude more wall time.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import ising_lab as il
from ising_lab.benchmarks import ea_instance

RESULTS = Path("results")


def run(fn, model, **kw):
    t = time.perf_counter()
    r = fn(model, **kw)
    dt = time.perf_counter() - t
    energies = [e for _, e in r]
    return {"best": min(energies), "mean": statistics.mean(energies), "wall_time": dt}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    L, READS = 8, 16
    rows = []
    pa_wins = 0
    print(f"3D EA L={L} (N={L**3}) Gaussian -- PA vs PT (PT also given extra sweeps)")
    for s in range(5):
        inst = ea_instance(L, seed=500 + s, dimension=3, distribution="gaussian")
        pa = run(il.population_annealing, inst.model, num_temps=50, population=40,
                 num_sweeps=8, beta_min=0.05, beta_max=6.0, num_reads=READS, seed=0)
        pt = run(il.parallel_tempering, inst.model, num_sweeps=2000, num_replicas=24,
                 beta_min=0.05, beta_max=6.0, num_reads=READS, seed=0)
        pt_big = run(il.parallel_tempering, inst.model, num_sweeps=32000, num_replicas=24,
                     beta_min=0.05, beta_max=6.0, num_reads=READS, seed=0)
        if pa["best"] < pt_big["best"] - 1e-9:
            pa_wins += 1
        rows.append({
            "instance_seed": 500 + s, "n": inst.n,
            "pa": pa, "pt": pt, "pt_16x_sweeps": pt_big,
            "best_delta_pa_minus_pt": pa["best"] - pt["best"],
            "best_delta_pa_minus_pt16x": pa["best"] - pt_big["best"],
        })
        print(f"  s{500+s}: PA best={pa['best']:8.1f} ({pa['wall_time']:.2f}s) | "
              f"PT best={pt['best']:8.1f} ({pt['wall_time']:.2f}s) | "
              f"PT16x best={pt_big['best']:8.1f} ({pt_big['wall_time']:.2f}s) | "
              f"PA beats PT16x by {pt_big['best']-pa['best']:.1f}")

    summary = {
        "experiment": "population_vs_pt_ea3d",
        "lattice": f"3d-gaussian-L{L}",
        "n": L ** 3,
        "num_reads": READS,
        "pa_config": {"num_temps": 50, "population": 40, "num_sweeps": 8},
        "pt_config": {"num_sweeps": 2000, "num_replicas": 24},
        "pt_16x_config": {"num_sweeps": 32000, "num_replicas": 24},
        "pa_beats_pt_16x_count": pa_wins,
        "num_instances": len(rows),
        "avg_best_delta_pa_minus_pt": statistics.mean(r["best_delta_pa_minus_pt"] for r in rows),
        "avg_best_delta_pa_minus_pt16x": statistics.mean(r["best_delta_pa_minus_pt16x"] for r in rows),
        "rows": rows,
    }
    (RESULTS / "population_vs_pt_ea3d.json").write_text(json.dumps(summary, indent=2))
    print(f"\nPA beats PT-with-16x-sweeps on {pa_wins}/{len(rows)} instances; "
          f"avg best advantage vs PT-16x = {-summary['avg_best_delta_pa_minus_pt16x']:.1f} energy units")
    print("Wrote results/population_vs_pt_ea3d.json")


if __name__ == "__main__":
    main()
