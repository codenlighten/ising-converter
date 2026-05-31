"""Loop-corrected belief propagation (Chertkov-Chernyak) vs plain Bethe BP.

Writes results/loop_bp_free_energy.json. On small loopy frustrated grids where
the exact free energy is brute-forceable, the truncated simple-cycle loop
correction is compared against the plain Bethe free energy. The correction is
exact on a single cycle (ring) and a large, systematic improvement on grids;
the gain shrinks as graphs get denser (dropped higher-order generalized loops).
"""
from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path

import ising_lab as il
from ising_lab.inference import loop_corrected_free_energy

RESULTS = Path("results")


def exact_free_energy(model, beta, n):
    z = sum(math.exp(-beta * model.energy(list(b))) for b in itertools.product((-1, 1), repeat=n))
    return -math.log(z) / beta


def grid(rows, cols, seed):
    rng = random.Random(seed)
    n = rows * cols
    edges = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                edges.append((i, i + 1, rng.choice([-1.0, 1.0])))
            if r + 1 < rows:
                edges.append((i, i + cols, rng.choice([-1.0, 1.0])))
    h = [0.5 * rng.uniform(-1, 1) for _ in range(n)]
    model = il.IsingModel(n, h, [(min(i, j), max(i, j), w) for i, j, w in edges])
    return model, n


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    beta = 0.6
    rows = []
    print("loop-corrected BP vs Bethe (exact free energy is brute-forced)")
    print(f"{'graph':>9} | {'bethe_err':>10} {'loop_err':>10} {'improvement':>11} {'cycles':>7}")
    # A ring first (single cycle -> exact).
    n = 8
    ring = il.IsingModel(n, [0.3] * n, [(i, (i + 1) % n, 0.8) for i in range(n)])
    fe = exact_free_energy(ring, beta, n)
    lf, bf, nc, conv = loop_corrected_free_energy(ring, beta, max_cycle_len=n)
    rows.append({"graph": f"ring-{n}", "n": n, "bethe_err": abs(bf - fe),
                 "loop_err": abs(lf - fe), "cycles": nc, "converged": conv})
    print(f"{'ring-'+str(n):>9} | {abs(bf-fe):10.2e} {abs(lf-fe):10.2e} "
          f"{abs(bf-fe)/max(abs(lf-fe),1e-300):10.0f}x {nc:7d}")
    for (r, c) in [(3, 3), (3, 4), (4, 4)]:
        for seed in range(3):
            model, n = grid(r, c, seed)
            fe = exact_free_energy(model, beta, n)
            lf, bf, nc, conv = loop_corrected_free_energy(model, beta, max_cycle_len=8)
            eb, el = abs(bf - fe), abs(lf - fe)
            rows.append({"graph": f"{r}x{c}-s{seed}", "n": n, "bethe_err": eb,
                         "loop_err": el, "cycles": nc, "converged": conv})
            print(f"{f'{r}x{c} s{seed}':>9} | {eb:10.2e} {el:10.2e} {eb/max(el,1e-300):10.1f}x {nc:7d}")

    summary = {
        "experiment": "loop_corrected_bp_free_energy",
        "beta": beta,
        "method": "Chertkov-Chernyak simple-cycle truncation (max_cycle_len=8)",
        "rows": rows,
        "mean_improvement_factor": sum(
            r["bethe_err"] / max(r["loop_err"], 1e-300) for r in rows
        ) / len(rows),
    }
    (RESULTS / "loop_bp_free_energy.json").write_text(json.dumps(summary, indent=2))
    print("\nWrote results/loop_bp_free_energy.json")


if __name__ == "__main__":
    main()
