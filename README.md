# ising_lab

[![CI](https://github.com/codenlighten/ising-converter/actions/workflows/ci.yml/badge.svg)](https://github.com/codenlighten/ising-converter/actions/workflows/ci.yml)

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
  - `parallel_tempering_houdayer` — PT with Houdayer isoenergetic cluster moves
  - `population_annealing` — sequential Monte Carlo with Boltzmann resampling
  - `population_annealing_icm` — PA + Houdayer cluster moves (strongest on 3D EA)
  - `belief_propagation` — deterministic sum-product inference (marginals, Bethe free energy)
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

## Cluster moves and the Parisi yardstick

Two physics-aware tools for the hard spin-glass regime (`scripts/bench_houdayer.py`):

- **Houdayer-PT** (`parallel_tempering_houdayer`) layers **isoenergetic cluster
  moves** on parallel tempering. It runs two replica lanes and flips connected
  clusters of disagreeing spins, tunnelling through barriers single-spin flips
  cannot cross. This is effective on **sparse / finite-dimensional** graphs —
  the 3D Edwards–Anderson lattice, which is also the regime of hardware
  spin-glass annealers. On a fully connected SK instance the disagreement graph
  percolates into one cluster and the move degenerates to a trivial global swap,
  so use plain `parallel_tempering` there. At matched compute on L=8 (N=512)
  Gaussian 3D EA, Houdayer-PT reaches lower mean energy on 8/8 benchmark
  instances (see `results/houdayer_vs_pt_ea3d.json`).

- **Population annealing** (`population_annealing`) carries a *population* of
  replicas and anneals β upward, **resampling** by the Boltzmann factor
  `exp(−Δβ·E)` at each step (low-energy replicas multiply, high-energy ones die)
  before equilibrating with Metropolis sweeps. On hard 3D EA Gaussian glasses it
  is the strongest sampler here by a wide margin: on L=8 (N=512) it reaches
  energies ~28 units lower than parallel tempering **even when PT is given 16×
  the sweeps** (~50× the wall time), winning on 5/5 benchmark instances
  (`results/population_vs_pt_ea3d.json`).

- **Population annealing + cluster moves** (`population_annealing_icm`) is the
  Wang–Machta–Katzgraber combination — Houdayer isoenergetic cluster moves
  between random pairs of the PA population at each temperature, the literature's
  strongest classical method for 3D EA glasses. Honest finding: it improves on
  plain PA but *incrementally* (~1–1.4 energy units typical at L=8, better on
  4/5 instances at matched compute — `results/pa_icm_vs_pa_ea3d.json`), because
  PA's resampling already does most of the work. A no-op on fully connected SK.

- **Belief propagation** (`belief_propagation`, `ising_lab.inference`) is a
  deterministic message-passing alternative to Monte Carlo. It returns spin
  marginals and the **Bethe free energy** — observables the samplers don't
  directly produce — **exact on trees** (validated to machine precision; a
  2000-node tree converges in ~0.05 s, where brute force would need 2²⁰⁰⁰
  evaluations), and the Bethe approximation on loopy graphs. Honest limitation:
  as a *ground-state* heuristic it is weak on frustrated loopy glasses — rounding
  its marginals on 3D EA lands far above population annealing (it may "converge"
  to a useless fixed point). That failure is exactly what loop-corrected BP and
  tensor-network methods exist to fix; BP here is the correct, fast inference
  core to build those on, not a 3D-glass optimizer.

- **Loop-corrected BP** (`loop_corrected_free_energy`) implements the
  Chertkov–Chernyak loop series: `Z = Z_Bethe · (1 + Σ over generalized loops)`,
  truncated at simple cycles, each cycle contributing
  `Π_{(ij)∈C} χ_ij / √((1−m_i²)(1−m_j²))`. It is **exact on a single cycle**
  (a ring — validated to machine precision) and a large, systematic improvement
  on loopy graphs: on brute-forceable 3×3–4×4 frustrated grids it cuts the
  free-energy error by ~3×–600× (`results/loop_bp_free_energy.json`). Honest
  limitation: the truncation drops higher-order generalized loops, so the gain
  shrinks with graph density and can slightly *overshoot* when Bethe is already
  near-exact — a principled leading-order correction, not a black box.

- **Parisi density as truth** (`sk_parisi_reference_energy`,
  `PARISI_SK_ENERGY_DENSITY`). The SK ground-state energy density converges to
  the analytically known Parisi constant, `E₀/N → −0.7632`, i.e.
  `E₀ ≈ −0.7632·N^{3/2}` for these un-normalized couplings. This gives an
  **absolute** large-N yardstick where brute force is hopeless — finite systems
  sit just above it by an O(N^{−2/3}) finite-size correction
  (`results/sk_parisi_convergence.json`).

```python
import ising_lab as il
from ising_lab.benchmarks import ea_instance, sk_instance, sk_energy_density

# Cluster-move PT on a 3D EA lattice
inst = ea_instance(8, seed=0, dimension=3, distribution="gaussian")
res = il.parallel_tempering_houdayer(
    inst.model, num_sweeps=2000, num_replicas=24, icm_every=5, num_reads=16, seed=0,
)

# How close is a sample to the Parisi thermodynamic-limit density?
sk = sk_instance(400, seed=0, distribution="gaussian")
r = il.parallel_tempering(sk.model, num_sweeps=8000, num_replicas=24, num_reads=16, seed=0)
print(sk_energy_density(min(e for _, e in r), 400))   # -> approaches -0.7632
```

## Results: method comparison

A unified benchmark across every method (`scripts/bench_all_methods.py`,
artifact in `results/all_methods_comparison.json`). The honest summary: the best
method depends on the regime, and the gaps grow with problem hardness.

**Sherrington–Kirkpatrick, exact truth** (success probability / median TTS₉₉).
All replica methods solve these; population annealing has the best
time-to-solution.

| method        | N=16 success | N=16 TTS₉₉ | N=24 success | N=24 TTS₉₉ |
|---------------|:------------:|:----------:|:------------:|:----------:|
| SA            | 96 %         | 0.0001 s   | 96 %         | 0.0003 s   |
| PT            | 100 %        | 0.0018 s   | 100 %        | 0.0033 s   |
| Houdayer-PT   | 100 %        | 0.0041 s   | 100 %        | 0.0074 s   |
| **PA**        | **100 %**    | **0.0007 s** | **100 %**  | **0.0012 s** |

**3D Edwards–Anderson, Gaussian, L=6 (N=216)** — the sparse-lattice regime of
hardware annealers (mean best / mean typical energy / wall time, 5 instances):

| method        | mean best | mean typical | wall   |
|---------------|:---------:|:------------:|:------:|
| SA            | −363.3    | −359.0       | 0.04 s |
| PT            | −356.1    | −352.9       | 0.53 s |
| Houdayer-PT   | −357.5    | −354.4       | 1.32 s |
| **PA**        | **−363.6**| **−363.1**   | 0.17 s |
| BP (baseline) | +16.9     | —            | 0.00 s |

PA wins on both best and (especially) typical-run energy; the margin over PT
widens at larger L (at L=8 it reaches energies ~28 units lower than PT given 16×
the sweeps — see `results/population_vs_pt_ea3d.json`). SA is competitive on
best-of at this smaller size; belief propagation, rounded to a configuration, is
not an optimizer here.

**Sherrington–Kirkpatrick, N=200** (beyond brute force; energy density vs the
Parisi constant −0.7632). All samplers tie at −0.7453 — a +0.018 gap that is the
expected O(N^{−2/3}) finite-size correction, not sampler error. On mean-field SK
the landscape is easy enough that method choice barely matters.

### Time-to-solution scaling (PA vs PT)

The rigorous version of "PA beats PT": optimal **work-to-solution** in
hardware-independent Monte Carlo sweep units (Rønnow-style — budget scanned, the
minimum taken), on 3D EA Gaussian, median over 5 instances per size
(`scripts/bench_tts_scaling.py`, `results/tts_scaling_ea3d.json`):

| L | N   | PA W\*    | PT W\*       | PT / PA |
|---|-----|----------:|-------------:|--------:|
| 3 | 27  | 1,200     | 16,000       | 13×     |
| 4 | 64  | 3,250     | 55,746       | 17×     |
| 5 | 125 | 8,764     | 11,189,425   | **1,277×** |
| 6 | 216 | 80,483    | *unreached*  | ∞       |

PA's work-to-solution grows gracefully with N; PT's diverges — a roughly
constant ~15× edge at small sizes blows up to >1000× by N=125, and by N=216 PT
fails to reach the best-known energy within the tested budget (≤256k sweep
units) while PA solves it. Sweep units remove the multicore-parallelism
confound, so this is an *algorithmic* separation, not a hardware artifact.

Honest caveats: (i) the scan optimizes over per-attempt *sweeps* at fixed
PA population/replica counts, so both W\* values are upper bounds on each
method's globally-optimal TTS; the robust conclusion is the *separation*, not
the absolute numbers. (ii) "Unreached" means PT did not reach the
**heavy-PA-derived** best-known energy (validated to equal brute force only at
N≤27), not that PT cannot find the true ground state. (iii) PT's large-N W\*
rests on few successes per 60 reads, so it is noisy to a factor of ~2 — far
smaller than the gap.

## Competing with D-Wave?

D-Wave's strongest published *optimization* scaling-advantage claim (Munoz Bauza
& Lidar, *Phys. Rev. Lett.* 134, 160601, 2025) is measured **against PT-ICM** —
parallel tempering with isoenergetic cluster moves, i.e. this lab's
`parallel_tempering_houdayer`. We independently find that **population annealing
beats PT-ICM by ~6× at N=64, ~1,800× at N=125, and unboundedly by N=216** on 3D
EA (work-to-solution, `results/pa_vs_pticm_tts_ea3d.json`). That motivates a
falsifiable hypothesis — a PA baseline could erode the reported QA advantage —
but it is **not** a refutation: their instance class (2D degree-5, high-precision)
and metric (wall-clock time-to-epsilon) differ from ours, and we have not run
their benchmark. The precise claim, caveats, and a roadmap to an honest
head-to-head are in [docs/DWAVE_COMPARISON.md](docs/DWAVE_COMPARISON.md).

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
