"""Houdayer isoenergetic-cluster-move PT vs plain PT, and the Parisi SK yardstick.

Two experiments, each writing a JSON artifact to results/:

1. houdayer_vs_pt_ea3d.json
   3D Edwards-Anderson (sparse lattice = the regime of hardware spin-glass
   annealers). Houdayer-PT runs two replica lanes, so it costs ~2x the
   spin-flips of plain PT per read; we give plain PT 2x the reads to match
   total compute. Metric: mean energy per read (typical-run quality) -- ICM
   improves equilibration, which shows up in the typical run more than in the
   best-of-many. We report a per-instance sign test.

2. sk_parisi_convergence.json
   Sherrington-Kirkpatrick (fully connected). Houdayer is a no-op here, so we
   just sample strongly and show the best energy *density* converging to the
   analytically known Parisi value -0.7632 from finite size -- using the
   universal constant as an absolute truth where brute force is impossible.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import ising_lab as il
from ising_lab.benchmarks import (
    PARISI_SK_ENERGY_DENSITY,
    ea_instance,
    sk_energy_density,
    sk_instance,
    sk_parisi_reference_energy,
)

RESULTS = Path("results")


def houdayer_vs_pt_ea3d(
    L: int = 8,
    num_instances: int = 8,
    num_replicas: int = 24,
    num_sweeps: int = 2000,
    reads: int = 16,
    icm_every: int = 5,
    beta_min: float = 0.05,
    beta_max: float = 6.0,
) -> dict:
    rows = []
    hou_better = 0
    for s in range(num_instances):
        inst = ea_instance(L, seed=500 + s, dimension=3, distribution="gaussian")
        # Plain PT gets 2x reads to match Houdayer's two lanes.
        pt = il.parallel_tempering(
            inst.model, num_sweeps=num_sweeps, num_replicas=num_replicas,
            beta_min=beta_min, beta_max=beta_max, num_reads=2 * reads, seed=0,
        )
        hou = il.parallel_tempering_houdayer(
            inst.model, num_sweeps=num_sweeps, num_replicas=num_replicas,
            beta_min=beta_min, beta_max=beta_max, icm_every=icm_every,
            num_reads=reads, seed=0,
        )
        e_pt = [e for _, e in pt]
        e_hou = [e for _, e in hou]
        d_mean = statistics.mean(e_hou) - statistics.mean(e_pt)
        if d_mean < 0:
            hou_better += 1
        rows.append({
            "instance_seed": 500 + s,
            "n": inst.n,
            "pt_mean": statistics.mean(e_pt),
            "pt_min": min(e_pt),
            "hou_mean": statistics.mean(e_hou),
            "hou_min": min(e_hou),
            "mean_delta_hou_minus_pt": d_mean,
            "min_delta_hou_minus_pt": min(e_hou) - min(e_pt),
        })
        print(f"  L{L} s{500 + s}: meanΔ={d_mean:+6.2f}  "
              f"minΔ={min(e_hou) - min(e_pt):+6.2f}  {'HOU' if d_mean < 0 else 'PT'}")
    summary = {
        "experiment": "houdayer_vs_pt_ea3d",
        "lattice": f"3d-gaussian-L{L}",
        "n": L ** 3,
        "num_replicas": num_replicas,
        "num_sweeps": num_sweeps,
        "houdayer_reads": reads,
        "pt_reads": 2 * reads,
        "icm_every": icm_every,
        "houdayer_better_mean_count": hou_better,
        "num_instances": num_instances,
        "avg_mean_delta": statistics.mean(r["mean_delta_hou_minus_pt"] for r in rows),
        "avg_min_delta": statistics.mean(r["min_delta_hou_minus_pt"] for r in rows),
        "rows": rows,
    }
    print(f"  -> Houdayer lower mean energy on {hou_better}/{num_instances}, "
          f"avg meanΔ={summary['avg_mean_delta']:+.2f}")
    return summary


def sk_parisi_convergence(
    sizes=(50, 100, 200, 400),
    instances_per_size: int = 3,
    num_replicas: int = 24,
    num_sweeps: int = 8000,
    reads: int = 16,
) -> dict:
    rows = []
    for n in sizes:
        densities = []
        for s in range(instances_per_size):
            inst = sk_instance(n, seed=900 + s, distribution="gaussian")
            r = il.parallel_tempering(
                inst.model, num_sweeps=num_sweeps, num_replicas=num_replicas,
                beta_min=0.05, beta_max=6.0, num_reads=reads, seed=0,
            )
            densities.append(sk_energy_density(min(e for _, e in r), n))
        e_density = statistics.mean(densities)
        rows.append({
            "n": n,
            "energy_density": e_density,
            "gap_to_parisi": e_density + PARISI_SK_ENERGY_DENSITY,
            "parisi_reference_energy": sk_parisi_reference_energy(n),
        })
        print(f"  N={n:>4}: e={e_density:.4f}  gap_to_parisi={e_density + PARISI_SK_ENERGY_DENSITY:+.4f}")
    return {
        "experiment": "sk_parisi_convergence",
        "parisi_density": -PARISI_SK_ENERGY_DENSITY,
        "num_replicas": num_replicas,
        "num_sweeps": num_sweeps,
        "reads": reads,
        "rows": rows,
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print("=== Houdayer-PT vs PT on 3D EA (Gaussian, L=8) ===")
    ea = houdayer_vs_pt_ea3d()
    (RESULTS / "houdayer_vs_pt_ea3d.json").write_text(json.dumps(ea, indent=2))

    print("\n=== SK best energy density vs Parisi constant ===")
    sk = sk_parisi_convergence()
    (RESULTS / "sk_parisi_convergence.json").write_text(json.dumps(sk, indent=2))

    print("\nWrote results/houdayer_vs_pt_ea3d.json and results/sk_parisi_convergence.json")


if __name__ == "__main__":
    main()
