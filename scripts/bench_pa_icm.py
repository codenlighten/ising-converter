"""Population annealing with Houdayer cluster moves (PA+ICM) vs plain PA.

Writes results/pa_icm_vs_pa_ea3d.json. PA+ICM (Wang-Machta-Katzgraber) applies
isoenergetic cluster moves between random pairs of the population at each
temperature -- the strongest classical method for 3D Edwards-Anderson glasses.
The honest finding: it improves on plain PA, but incrementally, since PA's
resampling already does most of the work.
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
    e = [x for _, x in r]
    return min(e), statistics.mean(e), dt


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    L, READS = 8, 16
    rows = []
    print(f"3D EA L={L} (N={L**3}) Gaussian: plain PA vs PA+ICM (matched temps/pop/sweeps)")
    print(f"{'inst':>9} | {'PA best':>9} {'ICM best':>9} | {'PA mean':>9} {'ICM mean':>9} | bestD  meanD")
    for s in range(5):
        inst = ea_instance(L, seed=500 + s, dimension=3, distribution="gaussian")
        pa = run(il.population_annealing, inst.model, num_temps=50, population=40,
                 num_sweeps=8, beta_min=0.05, beta_max=6.0, num_reads=READS, seed=0)
        icm = run(il.population_annealing_icm, inst.model, num_temps=50, population=40,
                  num_sweeps=8, beta_min=0.05, beta_max=6.0, icm_every=1, num_reads=READS, seed=0)
        rows.append({
            "instance_seed": 500 + s, "n": inst.n,
            "pa_best": pa[0], "pa_mean": pa[1], "pa_wall": pa[2],
            "icm_best": icm[0], "icm_mean": icm[1], "icm_wall": icm[2],
            "best_delta_icm_minus_pa": icm[0] - pa[0],
            "mean_delta_icm_minus_pa": icm[1] - pa[1],
        })
        print(f"{'L8 s'+str(500+s):>9} | {pa[0]:9.1f} {icm[0]:9.1f} | {pa[1]:9.1f} {icm[1]:9.1f} | "
              f"{icm[0]-pa[0]:+5.1f} {icm[1]-pa[1]:+5.1f}")

    summary = {
        "experiment": "pa_icm_vs_pa_ea3d",
        "lattice": f"3d-gaussian-L{L}",
        "n": L ** 3,
        "num_reads": READS,
        "config": {"num_temps": 50, "population": 40, "num_sweeps": 8, "icm_every": 1},
        "rows": rows,
        "mean_best_delta": statistics.mean(r["best_delta_icm_minus_pa"] for r in rows),
        "mean_mean_delta": statistics.mean(r["mean_delta_icm_minus_pa"] for r in rows),
        "icm_better_best_count": sum(1 for r in rows if r["best_delta_icm_minus_pa"] < -1e-9),
        "icm_better_mean_count": sum(1 for r in rows if r["mean_delta_icm_minus_pa"] < 0),
    }
    (RESULTS / "pa_icm_vs_pa_ea3d.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmean bestD={summary['mean_best_delta']:+.2f}  meanD={summary['mean_mean_delta']:+.2f} "
          f"(negative = ICM better); ICM better on mean {summary['icm_better_mean_count']}/5")
    print("Wrote results/pa_icm_vs_pa_ea3d.json")


if __name__ == "__main__":
    main()
