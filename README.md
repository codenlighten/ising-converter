# ising_lab

A Python lab for converting combinatorial problems into **Ising / QUBO** form and
solving them with a fast **Rust** kernel (simulated annealing and parallel
tempering), backed by a benchmark harness, a best-known-optimum registry, and
[dimod](https://docs.ocean.dwavesys.com/) interop.

The Ising energy is

```
H(s) = sum_i h_i s_i + sum_{(i,j) in edges} J_ij s_i s_j,    s_i in {-1, +1}
```

and the QUBO form is the binary (`x_i in {0, 1}`) equivalent, related by
`x_i = (1 + s_i) / 2`.

## Features

- **Rust kernel** (`src/lib.rs`, exposed as `ising_lab._kernel`)
  - `simulated_anneal` — geometric beta schedule, parallel reads via rayon
  - `parallel_tempering` / `parallel_tempering_diagnostic` / `parallel_tempering_with_betas`
  - `brute_force_ground_state` / `brute_force_min_energy` (exact, N ≤ 30)
  - Deterministic for a fixed `seed`, regardless of thread scheduling.
- **Problem encoders** (`ising_lab.problems`) — reference implementations of the
  recipes in Lucas (2014), *Ising formulations of many NP problems*: max-cut,
  number partitioning, TSP, graph coloring, knapsack, cardinality constraints.
- **QUBO model** (`ising_lab.QUBO`, `qubo_to_ising`) with an exact energy check.
- **Benchmark harness** (`ising_lab.benchmarks`) — Sherrington–Kirkpatrick (SK)
  and Edwards–Anderson (EA) spin-glass suites, time-to-solution (TTS) metrics,
  CSV/JSON export, and beta-ladder auto-tuners (equal-acceptance and
  Katzgraber–Trebst–Huse–Troyer feedback-optimized).
- **Optimum registry** (`ising_lab.OptimumRegistry`) — a monotone-in-energy,
  JSON-persisted record of the best solution known per instance.
- **dimod interop** (`ising_lab.dimod_adapter`) — convert any BQM to/from an
  `IsingModel`, and expose the SA/PT kernels as `dimod.Sampler` subclasses.

## Install

The project is built with [maturin](https://www.maturin.rs/). Requires Python
≥ 3.9 and a Rust toolchain.

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release          # build the Rust kernel and install editable

# optional extras
pip install -e ".[dev]"            # pytest + maturin + dimod
pip install -e ".[dimod]"          # dimod only (for the interop layer)
```

## Quickstart

```python
import ising_lab as il
from ising_lab.problems import max_cut
from ising_lab.qubo import spins_to_bits

# Max-cut on a 4-cycle. Returns (model, offset); cut_size = -(energy + offset).
model, offset = max_cut(4, [(0, 1), (1, 2), (2, 3), (3, 0)])

results = il.simulated_anneal(model, num_sweeps=500, num_reads=20, seed=7)
state, energy = min(results, key=lambda r: r[1])
print("cut size:", -(energy + offset))   # -> 4.0
print("partition:", spins_to_bits(state))
```

Parallel tempering on a frustrated instance:

```python
results = il.parallel_tempering(
    model, num_sweeps=1000, num_replicas=8,
    beta_min=0.1, beta_max=10.0, num_reads=10, seed=42,
)
best_energy = min(e for _, e in results)
```

## Benchmarking

```python
from ising_lab.benchmarks import sk_suite, benchmark, wrap_sa, wrap_pt, records_to_csv
from ising_lab import OptimumRegistry

instances = sk_suite(sizes=[12, 16], instances_per_size=5, distribution="binary")
reg = OptimumRegistry("results/sk_registry.json")

records = benchmark(
    {"sa": wrap_sa(num_sweeps=1000), "pt": wrap_pt(num_sweeps=1000)},
    instances,
    num_reads=50,
    registry=reg,        # seeds/records best-known truth; brute-forces N <= 30
)
reg.save()
records_to_csv(records, "results/sk_sa_vs_pt.csv")
```

Each `BenchmarkRecord` carries best/median/mean energy, wall time, success
probability against ground truth, and `tts_99` (time-to-solution at 99%
confidence). See `scripts/` for the benchmark runs whose outputs live in
`results/`.

## dimod interop

```python
import dimod
from ising_lab.dimod_adapter import SimulatedAnnealingSampler, to_bqm

sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample(my_bqm, num_sweeps=1000, num_reads=100, seed=1)

# or wrap an external dimod sampler (e.g. neal) into the benchmark harness:
from ising_lab.benchmarks import wrap_dimod
```

## Development

```bash
maturin develop --release    # rebuild after editing src/lib.rs
pytest -q                    # 52 tests
cargo clippy --release       # lint the kernel (clean)
```

## Layout

```
src/lib.rs                 Rust kernel (SA, PT, brute force, IsingModel)
python/ising_lab/          Python package
  qubo.py                  QUBO model + Ising conversion
  problems.py              combinatorial problem encoders
  benchmarks.py            SK/EA suites, harness, beta-ladder tuners
  registry.py              best-known-optimum registry
  dimod_adapter.py         BQM <-> IsingModel, dimod.Sampler wrappers
tests/                     pytest suite
scripts/                   benchmark runners
results/                   benchmark outputs (CSV/JSON)
```

## References

- A. Lucas, *Ising formulations of many NP problems*, Front. Phys. 2 (2014).
- Katzgraber, Trebst, Huse, Troyer, *Feedback-optimized parallel tempering Monte
  Carlo*, Phys. Rev. E 73, 056704 (2006).

## License

MIT — see [LICENSE](LICENSE).
