"""Benchmark harness for Sherrington-Kirkpatrick (SK) spin-glass instances.

The SK model is the canonical hard benchmark for Ising solvers:
fully connected graph on N spins, random couplings, no fields. Even at modest
N (50-100) the landscape has exponentially many local minima and exact methods
become infeasible -- which is exactly why it's used to compare SA, PT, QPU,
neal, and other samplers.

Sampler callable contract:
    sampler(model: IsingModel, num_reads: int) -> list[tuple[list[int], float]]

Wrap any ising_lab kernel or dimod.Sampler via `wrap_sa`, `wrap_pt`, or
`wrap_dimod`. Mix-and-match them in a single `benchmark()` call to A/B
sampler stacks on the same instances.
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Union

from ._kernel import (
    IsingModel,
    brute_force_ground_state,
    parallel_tempering,
    parallel_tempering_diagnostic,
    parallel_tempering_houdayer,
    parallel_tempering_with_betas,
    population_annealing,
    population_annealing_icm,
    simulated_anneal,
)
from .registry import OptimumRegistry, sk_instance_key

# Parisi ground-state energy density of the Sherrington-Kirkpatrick model:
#   lim_{N->inf} E_0 / N  =  -0.763166... for the canonically normalized SK
#   Hamiltonian H = (1/sqrt(N)) sum_{i<j} J_ij s_i s_j with Var(J_ij) = 1.
# Source: Parisi RSB solution; numerical value e.g. Crisanti-Rizzo (2002).
PARISI_SK_ENERGY_DENSITY = 0.7631667

SamplerFn = Callable[[IsingModel, int], List[tuple]]
PathLike = Union[str, Path]

CSV_FIELDS = (
    "sampler",
    "instance_seed",
    "n",
    "num_reads",
    "best_energy",
    "median_energy",
    "mean_energy",
    "wall_time",
    "success_count",
    "success_prob",
    "tts_99",
    "ground_state_energy",
)


def pt_beta_ladder(beta_min: float, beta_max: float, num_replicas: int) -> List[float]:
    """The geometric (linear-in-log) beta ladder used internally by PT.

    Useful for interpreting swap-acceptance diagnostics: pair index k spans
    [betas[k], betas[k+1]], and a healthy ladder has acceptance ~0.2-0.4
    on every pair.
    """
    if num_replicas < 2:
        raise ValueError("num_replicas must be >= 2")
    if beta_min <= 0.0 or beta_max <= beta_min:
        raise ValueError("require 0 < beta_min < beta_max")
    log_lo = math.log(beta_min)
    log_hi = math.log(beta_max)
    denom = num_replicas - 1
    return [math.exp(log_lo + (log_hi - log_lo) * (k / denom)) for k in range(num_replicas)]


def sk_energy_density(energy: float, n: int) -> float:
    """Canonical SK energy density of a raw energy on one of our SK instances.

    Our SK couplings are un-normalized (Var(J_ij) = 1), so the raw energy scales
    as N^{3/2} relative to the canonically normalized SK Hamiltonian. Dividing by
    N^{3/2} yields a quantity directly comparable to the Parisi density
    `-PARISI_SK_ENERGY_DENSITY` in the thermodynamic limit.
    """
    return energy / (n ** 1.5)


def sk_parisi_reference_energy(n: int) -> float:
    """Asymptotic Parisi ground-state energy for one of our un-normalized SK
    instances: `-PARISI_SK_ENERGY_DENSITY * N^{3/2}`.

    This is the N -> infinity yardstick, not the exact finite-N ground state:
    finite systems sit *above* it by an O(N^{-2/3}) finite-size correction. Use
    it as an absolute reference scale at large N (where brute force is hopeless),
    e.g. to report `energy / reference` (-> 1 from above) or the density gap to
    the thermodynamic limit. By SK universality it applies to both the binary
    (+/-1) and Gaussian instances, whose couplings share Var(J_ij) = 1.
    """
    return -PARISI_SK_ENERGY_DENSITY * (n ** 1.5)


def parallel_tempering_with_diagnostics(
    model: IsingModel,
    num_sweeps: int = 1000,
    num_replicas: int = 8,
    beta_min: float = 0.1,
    beta_max: float = 10.0,
    swap_every: int = 1,
    num_reads: int = 1,
    seed: Optional[int] = None,
) -> List[dict]:
    """Run PT and return a per-read dict with results + diagnostics.

    Each dict has keys:
        state             -- best state seen by this chain (list of +/-1)
        energy            -- its energy
        swap_acceptance   -- list[float], one rate per adjacent replica pair
        final_energies    -- list[float], final energy at each beta position
    """
    raw = parallel_tempering_diagnostic(
        model,
        num_sweeps=num_sweeps,
        num_replicas=num_replicas,
        beta_min=beta_min,
        beta_max=beta_max,
        swap_every=swap_every,
        num_reads=num_reads,
        seed=seed,
    )
    return [
        {
            "state": state,
            "energy": energy,
            "swap_acceptance": swap_rates,
            "final_energies": final_energies,
            "round_trips": round_trips,
            "n_up": n_up,
            "n_down": n_down,
        }
        for state, energy, swap_rates, final_energies, round_trips, n_up, n_down in raw
    ]


def parallel_tempering_betas(
    model: IsingModel,
    betas: Sequence[float],
    num_sweeps: int = 1000,
    swap_every: int = 1,
    num_reads: int = 1,
    seed: Optional[int] = None,
) -> List[dict]:
    """Run PT with an explicit (non-geometric) beta ladder. Returns diagnostic dicts.

    Use this when you've tuned a custom ladder (e.g. via `auto_tune_beta_ladder`)
    and want to keep PT running on it.
    """
    raw = parallel_tempering_with_betas(
        model,
        list(betas),
        num_sweeps=num_sweeps,
        swap_every=swap_every,
        num_reads=num_reads,
        seed=seed,
    )
    return [
        {
            "state": state,
            "energy": energy,
            "swap_acceptance": swap_rates,
            "final_energies": final_energies,
            "round_trips": round_trips,
            "n_up": n_up,
            "n_down": n_down,
        }
        for state, energy, swap_rates, final_energies, round_trips, n_up, n_down in raw
    ]


def auto_tune_beta_ladder(
    model: IsingModel,
    num_replicas: int,
    beta_min: float,
    beta_max: float,
    target_acceptance: float = 0.3,
    pilot_sweeps: int = 500,
    pilot_reads: int = 4,
    num_iterations: int = 8,
    tolerance: float = 0.05,
    damping: float = 0.5,
    seed: Optional[int] = None,
) -> tuple:
    """Iteratively reshape a geometric beta ladder so every adjacent pair
    achieves swap acceptance near `target_acceptance`.

    Algorithm: start with a geometric ladder. Each iteration:
        1. Run a short PT pilot, average per-pair acceptance across reads.
        2. For each gap, scale log-spacing by (log(target)/log(a))**damping,
           clipped to [0.25, 4.0] per step.
        3. Renormalize so endpoints stay fixed.
    Convergence: every pair within `tolerance` of `target_acceptance`, or
    after `num_iterations`.

    Endpoints `beta_min` and `beta_max` are preserved exactly.

    Returns:
        final_betas (list[float], length num_replicas, strictly increasing)
        history    (list[list[float]], per-iteration average acceptance rates;
                    each inner list has length num_replicas - 1)
    """
    if num_replicas < 2:
        raise ValueError("num_replicas must be >= 2")
    if not 0.0 < target_acceptance < 1.0:
        raise ValueError("target_acceptance must be in (0, 1)")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in (0, 1]")

    betas = pt_beta_ladder(beta_min, beta_max, num_replicas)
    log_target = math.log(target_acceptance)
    history: list[list[float]] = []

    for it in range(num_iterations):
        diags = parallel_tempering_betas(
            model,
            betas=betas,
            num_sweeps=pilot_sweeps,
            num_reads=pilot_reads,
            seed=None if seed is None else seed + it,
        )
        avg_a = [
            statistics.mean(d["swap_acceptance"][k] for d in diags)
            for k in range(num_replicas - 1)
        ]
        history.append(avg_a)

        if all(abs(a - target_acceptance) <= tolerance for a in avg_a):
            break

        log_betas = [math.log(b) for b in betas]
        log_gaps = [log_betas[k + 1] - log_betas[k] for k in range(num_replicas - 1)]

        new_gaps: list[float] = []
        for k, a in enumerate(avg_a):
            a_clamped = min(max(a, 1e-6), 0.999)
            ratio = log_target / math.log(a_clamped)
            ratio = max(0.25, min(4.0, ratio))
            new_gaps.append(log_gaps[k] * (ratio ** damping))

        total_log_range = log_betas[-1] - log_betas[0]
        total_new = sum(new_gaps)
        if total_new <= 0.0:
            break  # numerical pathology; bail out with current ladder
        new_gaps = [g * total_log_range / total_new for g in new_gaps]

        new_log_betas = [log_betas[0]]
        for g in new_gaps:
            new_log_betas.append(new_log_betas[-1] + g)
        betas = [math.exp(x) for x in new_log_betas]

    return betas, history


def auto_tune_beta_ladder_ktht(
    model: IsingModel,
    num_replicas: int,
    beta_min: float,
    beta_max: float,
    pilot_sweeps: int = 2000,
    pilot_reads: int = 4,
    num_iterations: int = 5,
    swap_every: int = 1,
    seed: Optional[int] = None,
) -> tuple:
    """Katzgraber-Trebst-Huse-Troyer feedback-optimized beta-ladder tuner.

    Algorithm (Phys. Rev. E 73, 056704, 2006):
        f(beta) = n_up(beta) / (n_up(beta) + n_down(beta))
    where a replica is "up" if it last visited beta_min (heading to beta_max)
    and "down" otherwise. At each pilot:
        1. Aggregate n_up[k], n_down[k] across reads.
        2. Compute f[k]; force monotone-decreasing (running min from hot end).
        3. New positions solve f(beta_k_new) = 1 - k/(R-1), interpolated in
           log-beta space. Where f drops fastest, replicas pile up; where f
           is flat, they thin out -- which maximizes round-trip rate.

    Endpoints beta_min, beta_max are preserved. Requires meaningful round-trip
    statistics during the pilot; uses longer default pilots than the equal-
    acceptance tuner because flux estimates need replicas to actually traverse.

    Returns:
        final_betas (list[float], length num_replicas, strictly increasing)
        history     (list[list[float]], per-iteration f arrays)
    """
    if num_replicas < 2:
        raise ValueError("num_replicas must be >= 2")
    if beta_min <= 0.0 or beta_max <= beta_min:
        raise ValueError("require 0 < beta_min < beta_max")

    R = num_replicas
    betas = pt_beta_ladder(beta_min, beta_max, R)
    history: list[list[float]] = []

    for it in range(num_iterations):
        diags = parallel_tempering_betas(
            model,
            betas=betas,
            num_sweeps=pilot_sweeps,
            swap_every=swap_every,
            num_reads=pilot_reads,
            seed=None if seed is None else seed + it,
        )

        n_up_total = [0] * R
        n_down_total = [0] * R
        for d in diags:
            for k in range(R):
                n_up_total[k] += d["n_up"][k]
                n_down_total[k] += d["n_down"][k]

        f: list[float] = []
        for k in range(R):
            total = n_up_total[k] + n_down_total[k]
            f.append(n_up_total[k] / total if total > 0 else 0.5)
        history.append(f)

        # Force monotone-decreasing and anchor endpoints (1 at hot, 0 at cold).
        f_mono = list(f)
        for k in range(1, R):
            if f_mono[k] > f_mono[k - 1]:
                f_mono[k] = f_mono[k - 1]
        f_mono[0] = 1.0
        f_mono[-1] = 0.0

        if f_mono[0] - f_mono[-1] <= 0.0:
            break

        log_betas = [math.log(b) for b in betas]
        new_log_betas = [log_betas[0]]
        for k in range(1, R - 1):
            t = 1.0 - k / (R - 1)
            j = 0
            while j < R - 1 and f_mono[j + 1] > t:
                j += 1
            if j >= R - 1:
                new_log_betas.append(log_betas[-1])
                continue
            denom = f_mono[j] - f_mono[j + 1]
            if denom <= 0.0:
                new_log_betas.append(log_betas[j])
            else:
                alpha = (f_mono[j] - t) / denom
                new_log_betas.append(
                    log_betas[j] + alpha * (log_betas[j + 1] - log_betas[j])
                )
        new_log_betas.append(log_betas[-1])

        if not all(new_log_betas[k + 1] > new_log_betas[k] for k in range(R - 1)):
            break
        betas = [math.exp(x) for x in new_log_betas]

    return betas, history


def wrap_pt_ktht(
    num_sweeps: int = 1000,
    num_replicas: int = 8,
    beta_min: float = 0.1,
    beta_max: float = 10.0,
    pilot_sweeps: int = 2000,
    pilot_reads: int = 4,
    num_iterations: int = 5,
    swap_every: int = 1,
    seed: Optional[int] = None,
) -> SamplerFn:
    """Build a PT sampler that tunes its beta ladder via KTHT per instance.

    Pilot cost is charged to wall_time so TTS comparisons stay fair.
    """

    def _run(model: IsingModel, num_reads: int) -> List[tuple]:
        tuned_betas, _ = auto_tune_beta_ladder_ktht(
            model,
            num_replicas=num_replicas,
            beta_min=beta_min,
            beta_max=beta_max,
            pilot_sweeps=pilot_sweeps,
            pilot_reads=pilot_reads,
            num_iterations=num_iterations,
            swap_every=swap_every,
            seed=seed,
        )
        raw = parallel_tempering_with_betas(
            model,
            tuned_betas,
            num_sweeps=num_sweeps,
            swap_every=swap_every,
            num_reads=num_reads,
            seed=seed,
        )
        return [(state, energy) for state, energy, _, _, _, _, _ in raw]

    return _run


@dataclass
class SKInstance:
    """A Sherrington-Kirkpatrick instance plus its bookkeeping."""

    n: int
    seed: int
    distribution: str
    model: IsingModel
    ground_state_energy: Optional[float] = None  # populated by `solve_ground_state`


def sk_instance(
    n: int, seed: int, distribution: str = "binary"
) -> SKInstance:
    """Generate a reproducible SK instance.

    Fully connected graph (N(N-1)/2 couplings), no magnetic field.

    distribution:
        "binary"   -- J_ij in {-1, +1} uniformly (the +/-J SK model)
        "gaussian" -- J_ij ~ N(0, 1) (Gaussian SK; un-normalized)
    """
    rng = random.Random(seed)
    couplings: list[tuple[int, int, float]] = []
    if distribution == "binary":
        for i in range(n):
            for j in range(i + 1, n):
                couplings.append((i, j, 1.0 if rng.random() < 0.5 else -1.0))
    elif distribution == "gaussian":
        for i in range(n):
            for j in range(i + 1, n):
                couplings.append((i, j, rng.gauss(0.0, 1.0)))
    else:
        raise ValueError(f"unknown distribution {distribution!r}")
    model = IsingModel(n, [0.0] * n, couplings)
    return SKInstance(n=n, seed=seed, distribution=distribution, model=model)


def sk_suite(
    sizes: Sequence[int],
    instances_per_size: int = 5,
    distribution: str = "binary",
    base_seed: int = 0,
) -> List[SKInstance]:
    """Generate a benchmark suite: `instances_per_size` instances at each N."""
    out: list[SKInstance] = []
    for n in sizes:
        for k in range(instances_per_size):
            out.append(sk_instance(n, seed=base_seed + 1000 * n + k, distribution=distribution))
    return out


def ea_instance(
    L: int,
    seed: int,
    dimension: int = 3,
    distribution: str = "binary",
    periodic: bool = True,
) -> SKInstance:
    """Edwards-Anderson spin glass on an L^d cubic lattice (the canonical
    benchmark used in quantum-annealing literature).

    L: side length per dimension.
    dimension: 2 (square lattice) or 3 (cubic lattice).
    distribution: "binary" (J in {-1, +1}) or "gaussian" (J ~ N(0, 1)).
    periodic: wrap nearest-neighbor edges across boundaries.

    Reuses SKInstance for benchmark-harness compatibility but tags
    `instance.distribution = "ea-<dim>d-<dist>-L<L>"` so the registry key
    distinguishes EA instances from SK ones.
    """
    import itertools

    if dimension not in (2, 3):
        raise ValueError("only 2D and 3D supported")
    if L < 2:
        raise ValueError("L must be >= 2")
    if distribution not in ("binary", "gaussian"):
        raise ValueError(f"unknown distribution {distribution!r}")

    rng = random.Random(seed)
    n = L ** dimension

    def idx(coords):
        i = 0
        for c in coords:
            i = i * L + (c % L)
        return i

    edges_seen: set = set()
    couplings: list = []
    for coords in itertools.product(range(L), repeat=dimension):
        for d in range(dimension):
            neighbor = list(coords)
            neighbor[d] += 1
            if not periodic and neighbor[d] >= L:
                continue
            i, j = idx(coords), idx(neighbor)
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in edges_seen:
                continue
            edges_seen.add((a, b))
            w = (1.0 if rng.random() < 0.5 else -1.0) if distribution == "binary" else rng.gauss(0.0, 1.0)
            couplings.append((a, b, w))

    model = IsingModel(n, [0.0] * n, couplings)
    return SKInstance(
        n=n,
        seed=seed,
        distribution=f"ea-{dimension}d-{distribution}-L{L}",
        model=model,
    )


def ea_suite(
    L_values: Sequence[int],
    instances_per_L: int = 5,
    dimension: int = 3,
    distribution: str = "binary",
    base_seed: int = 0,
) -> List[SKInstance]:
    """Generate an EA benchmark suite: `instances_per_L` instances at each L."""
    out: list = []
    for L in L_values:
        n = L ** dimension
        for k in range(instances_per_L):
            out.append(ea_instance(
                L,
                seed=base_seed + 1000 * n + k,
                dimension=dimension,
                distribution=distribution,
            ))
    return out


def degree5_2d_instance(
    L: int,
    seed: int,
    distribution: str = "gaussian",
) -> SKInstance:
    """A 5-regular 2D toroidal spin glass -- a *proxy* for the 2D degree-5
    high-precision instance class on which D-Wave's strongest optimization
    scaling-advantage claim is benchmarked (Munoz Bauza & Lidar, PRL 134,
    160601, 2025). This is NOT their exact QAC graph; it matches the salient
    features: a planar/2D-local lattice of fixed degree 5 with continuous
    ("high-precision") couplings.

    Construction: an L x L square lattice with periodic boundaries (each site
    has 4 nearest neighbors) plus one diagonal bond per site, added by the parity
    rule "bond (x, y)-(x+1, y+1) iff x is even". Every site is the lower-left end
    of exactly one such diagonal (even columns) or the upper-right end of exactly
    one (odd columns), so the graph is exactly 5-regular. Requires L even, L >= 4.

    `distribution`: "gaussian" (J ~ N(0, 1), high precision) or "binary".
    """
    if L < 4 or L % 2 != 0:
        raise ValueError("L must be even and >= 4 for a clean 5-regular torus")
    if distribution not in ("gaussian", "binary"):
        raise ValueError(f"unknown distribution {distribution!r}")

    rng = random.Random(seed)
    n = L * L

    def idx(x: int, y: int) -> int:
        return (x % L) * L + (y % L)

    edges_seen: set = set()
    couplings: list = []

    def add(i: int, j: int) -> None:
        a, b = (i, j) if i < j else (j, i)
        if a == b or (a, b) in edges_seen:
            return
        edges_seen.add((a, b))
        w = rng.gauss(0.0, 1.0) if distribution == "gaussian" else (1.0 if rng.random() < 0.5 else -1.0)
        couplings.append((a, b, w))

    for x in range(L):
        for y in range(L):
            i = idx(x, y)
            add(i, idx(x + 1, y))   # nearest neighbors
            add(i, idx(x, y + 1))
            if x % 2 == 0:          # one diagonal per site (parity rule -> degree 5)
                add(i, idx(x + 1, y + 1))

    model = IsingModel(n, [0.0] * n, couplings)
    return SKInstance(n=n, seed=seed, distribution=f"deg5-2d-{distribution}-L{L}", model=model)


def solve_ground_state(instance: SKInstance) -> tuple:
    """Brute-force the ground state, cache its energy, return (state, energy). N <= 30."""
    state, energy = brute_force_ground_state(instance.model)
    instance.ground_state_energy = energy
    return state, energy


@dataclass
class BenchmarkRecord:
    sampler: str
    instance_seed: int
    n: int
    num_reads: int
    best_energy: float
    median_energy: float
    mean_energy: float
    wall_time: float
    success_count: int = 0
    success_prob: float = 0.0
    tts_99: Optional[float] = None  # time-to-99%-success in seconds, None if undefined
    ground_state_energy: Optional[float] = None
    energies: List[float] = field(default_factory=list, repr=False)


def _time_to_solution_99(p_success: float, time_per_read: float) -> Optional[float]:
    """Time-to-solution at 99% confidence.

    TTS_99 = time_per_read * log(1 - 0.99) / log(1 - p_success)
    where p_success is the per-read probability of finding a ground state.

    Returns:
        time_per_read         if p_success == 1
        None                  if p_success == 0 (can't estimate, effectively infinite)
        time_per_read * ratio otherwise
    """
    if p_success >= 1.0:
        return time_per_read
    if p_success <= 0.0:
        return None
    return time_per_read * math.log(1.0 - 0.99) / math.log(1.0 - p_success)


def benchmark(
    samplers: Mapping[str, SamplerFn],
    instances: Iterable[SKInstance],
    num_reads: int = 50,
    energy_tol: float = 1e-6,
    solve_truth: bool = True,
    registry: Optional[OptimumRegistry] = None,
) -> List[BenchmarkRecord]:
    """Run every sampler on every instance, return one record per (sampler, instance) pair.

    Ground-truth source (in priority order):
        1. registry.best(key).energy    -- if `registry` is passed and has an entry
        2. inst.ground_state_energy     -- if already cached
        3. brute force                  -- if `solve_truth=True` (requires N <= 30)
        4. None                         -- success stats left blank

    If `registry` is passed, every sampler's best result is offered to the
    registry. Strict improvements are recorded with `source=<sampler name>`,
    and the registry's notion of truth is used for subsequent instances in
    the same call (and in any future call sharing the registry).
    """
    records: list[BenchmarkRecord] = []
    instances = list(instances)

    for inst in instances:
        key = sk_instance_key(inst)

        # Seed truth from the registry if present.
        if registry is not None:
            cached = registry.best(key)
            if cached is not None and inst.ground_state_energy is None:
                inst.ground_state_energy = cached.energy

        # Fall back to brute force for small instances. Skip silently when N
        # is beyond the kernel's brute-force cap (currently 30).
        if inst.ground_state_energy is None and solve_truth and inst.n <= 30:
            state, energy = solve_ground_state(inst)
            if registry is not None:
                registry.update(
                    key,
                    energy=energy,
                    state=state,
                    source="brute_force",
                    metadata={"n": inst.n, "distribution": inst.distribution},
                )

        for name, sampler_fn in samplers.items():
            t0 = time.perf_counter()
            results = sampler_fn(inst.model, num_reads)
            wall_time = time.perf_counter() - t0
            energies = [float(e) for _, e in results]
            states = [list(s) for s, _ in results]

            success_count = 0
            success_prob = 0.0
            tts = None
            truth = inst.ground_state_energy
            if truth is not None:
                success_count = sum(1 for e in energies if e <= truth + energy_tol)
                success_prob = success_count / len(energies) if energies else 0.0
                if energies:
                    tts = _time_to_solution_99(success_prob, wall_time / len(energies))

            best_idx = min(range(len(energies)), key=lambda i: energies[i]) if energies else None
            best_energy = energies[best_idx] if best_idx is not None else float("inf")

            # Update the registry if this sampler beat the previous best.
            if registry is not None and best_idx is not None:
                improved = registry.update(
                    key,
                    energy=best_energy,
                    state=states[best_idx],
                    source=name,
                    metadata={"n": inst.n, "num_reads": num_reads, "wall_time": wall_time},
                )
                # If we strictly improved, the cached truth on the instance is stale.
                if improved:
                    inst.ground_state_energy = best_energy

            records.append(
                BenchmarkRecord(
                    sampler=name,
                    instance_seed=inst.seed,
                    n=inst.n,
                    num_reads=len(energies),
                    best_energy=best_energy,
                    median_energy=statistics.median(energies) if energies else float("nan"),
                    mean_energy=statistics.mean(energies) if energies else float("nan"),
                    wall_time=wall_time,
                    success_count=success_count,
                    success_prob=success_prob,
                    tts_99=tts,
                    ground_state_energy=truth,
                    energies=energies,
                )
            )
    return records


def wrap_sa(
    num_sweeps: int = 1000,
    beta_start: float = 0.1,
    beta_end: float = 10.0,
    seed: Optional[int] = None,
) -> SamplerFn:
    """Build a sampler callable for ising_lab.simulated_anneal with frozen hyperparams."""

    def _run(model: IsingModel, num_reads: int) -> List[tuple]:
        return simulated_anneal(
            model,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            beta_start=beta_start,
            beta_end=beta_end,
            seed=seed,
        )

    return _run


def wrap_pt(
    num_sweeps: int = 1000,
    num_replicas: int = 8,
    beta_min: float = 0.1,
    beta_max: float = 10.0,
    swap_every: int = 1,
    seed: Optional[int] = None,
) -> SamplerFn:
    """Build a sampler callable for ising_lab.parallel_tempering with frozen hyperparams."""

    def _run(model: IsingModel, num_reads: int) -> List[tuple]:
        return parallel_tempering(
            model,
            num_sweeps=num_sweeps,
            num_replicas=num_replicas,
            beta_min=beta_min,
            beta_max=beta_max,
            swap_every=swap_every,
            num_reads=num_reads,
            seed=seed,
        )

    return _run


def wrap_pt_houdayer(
    num_sweeps: int = 1000,
    num_replicas: int = 8,
    beta_min: float = 0.1,
    beta_max: float = 10.0,
    swap_every: int = 1,
    icm_every: int = 10,
    seed: Optional[int] = None,
) -> SamplerFn:
    """Build a sampler for `parallel_tempering_houdayer` with frozen hyperparams.

    Houdayer isoenergetic cluster moves (ICM) layered on PT. The move runs two
    replica lanes and tunnels through barriers via connected disagreement
    clusters -- effective on sparse / finite-dimensional graphs (e.g. the 3D
    Edwards-Anderson lattice, the regime of hardware spin-glass annealers). On a
    fully connected SK instance the clusters percolate and the move is a no-op,
    so prefer `wrap_pt` there. Note this runs 2x the replicas of plain PT per
    read; match `num_reads` accordingly for fair comparisons.
    """

    def _run(model: IsingModel, num_reads: int) -> List[tuple]:
        return parallel_tempering_houdayer(
            model,
            num_sweeps=num_sweeps,
            num_replicas=num_replicas,
            beta_min=beta_min,
            beta_max=beta_max,
            swap_every=swap_every,
            icm_every=icm_every,
            num_reads=num_reads,
            seed=seed,
        )

    return _run


def wrap_pa(
    num_temps: int = 30,
    population: int = 50,
    num_sweeps: int = 10,
    beta_min: float = 0.1,
    beta_max: float = 10.0,
    seed: Optional[int] = None,
) -> SamplerFn:
    """Build a sampler for `population_annealing` with frozen hyperparams.

    Each read is one independent PA run carrying a population of `population`
    replicas annealed through `num_temps` inverse-temperatures, resampling by
    Boltzmann weight at each step and equilibrating with `num_sweeps` Metropolis
    sweeps. On hard 3D Edwards-Anderson spin glasses PA reaches markedly lower
    energies than parallel tempering at equal or lower wall time
    (see scripts/bench_population.py).
    """

    def _run(model: IsingModel, num_reads: int) -> List[tuple]:
        return population_annealing(
            model,
            num_temps=num_temps,
            population=population,
            num_sweeps=num_sweeps,
            beta_min=beta_min,
            beta_max=beta_max,
            num_reads=num_reads,
            seed=seed,
        )

    return _run


def wrap_pa_icm(
    num_temps: int = 30,
    population: int = 50,
    num_sweeps: int = 10,
    beta_min: float = 0.1,
    beta_max: float = 10.0,
    icm_every: int = 1,
    seed: Optional[int] = None,
) -> SamplerFn:
    """Build a sampler for population annealing with Houdayer cluster moves.

    Applies isoenergetic cluster moves between random pairs of the population
    (all at one temperature per step) -- the Wang-Machta-Katzgraber combination,
    the strongest classical method for 3D Edwards-Anderson glasses. The gain
    over plain PA is incremental (PA's resampling already does most of the work)
    and shows mainly on sparse lattices; on fully connected SK the cluster moves
    are a no-op, so prefer `wrap_pa` there.
    """

    def _run(model: IsingModel, num_reads: int) -> List[tuple]:
        return population_annealing_icm(
            model,
            num_temps=num_temps,
            population=population,
            num_sweeps=num_sweeps,
            beta_min=beta_min,
            beta_max=beta_max,
            icm_every=icm_every,
            num_reads=num_reads,
            seed=seed,
        )

    return _run


def records_to_csv(
    records: Iterable["BenchmarkRecord"],
    path: PathLike,
) -> Path:
    """Write per-record summary rows to CSV (one row per record).

    Only scalar fields are written -- the per-read `energies` histogram is
    omitted. Use `records_to_json` for the full record. `None` values for
    `tts_99` and `ground_state_energy` are written as empty cells.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for rec in records:
            row = {k: getattr(rec, k) for k in CSV_FIELDS}
            writer.writerow(row)
    return out


def records_to_json(
    records: Iterable["BenchmarkRecord"],
    path: PathLike,
) -> Path:
    """Write full BenchmarkRecord rows (including the per-read `energies` list) as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in records], indent=2))
    return out


def records_from_json(path: PathLike) -> List["BenchmarkRecord"]:
    """Read a JSON file written by `records_to_json` back into BenchmarkRecord instances."""
    data = json.loads(Path(path).read_text())
    return [BenchmarkRecord(**d) for d in data]


def wrap_pt_tuned(
    num_sweeps: int = 1000,
    num_replicas: int = 8,
    beta_min: float = 0.1,
    beta_max: float = 10.0,
    target_acceptance: float = 0.3,
    pilot_sweeps: int = 500,
    pilot_reads: int = 4,
    num_iterations: int = 5,
    swap_every: int = 1,
    seed: Optional[int] = None,
) -> SamplerFn:
    """Build a PT sampler that auto-tunes the beta ladder per instance.

    The pilot cost (running short PT chains for ladder adjustment) is included
    in the sampler's wall time, so TTS comparisons stay fair against samplers
    that don't tune.
    """

    def _run(model: IsingModel, num_reads: int) -> List[tuple]:
        tuned_betas, _ = auto_tune_beta_ladder(
            model,
            num_replicas=num_replicas,
            beta_min=beta_min,
            beta_max=beta_max,
            target_acceptance=target_acceptance,
            pilot_sweeps=pilot_sweeps,
            pilot_reads=pilot_reads,
            num_iterations=num_iterations,
            seed=seed,
        )
        raw = parallel_tempering_with_betas(
            model,
            tuned_betas,
            num_sweeps=num_sweeps,
            swap_every=swap_every,
            num_reads=num_reads,
            seed=seed,
        )
        return [(state, energy) for state, energy, _, _, _, _, _ in raw]

    return _run


def wrap_dimod(dimod_sampler, **sample_kwargs) -> SamplerFn:
    """Wrap any dimod.Sampler (neal, DWaveSampler, ExactSolver, etc.) as a SamplerFn.

    Internally converts the IsingModel to a SPIN BQM via `to_bqm`, calls
    `dimod_sampler.sample(bqm, num_reads=..., **sample_kwargs)`, and converts
    the resulting SampleSet records back to (state, energy) tuples.
    """
    from .dimod_adapter import to_bqm  # lazy: dimod is an optional dep

    def _run(model: IsingModel, num_reads: int) -> List[tuple]:
        bqm = to_bqm(model)
        sampleset = dimod_sampler.sample(bqm, num_reads=num_reads, **sample_kwargs)
        out: list[tuple] = []
        variables = list(sampleset.variables)
        for record in sampleset.record:
            sample_dict = dict(zip(variables, record.sample))
            ordered = [int(sample_dict[i]) for i in range(model.num_spins)]
            out.append((ordered, float(record.energy)))
        return out

    return _run


__all__ = [
    "SKInstance",
    "sk_instance",
    "sk_suite",
    "ea_instance",
    "ea_suite",
    "degree5_2d_instance",
    "solve_ground_state",
    "BenchmarkRecord",
    "benchmark",
    "wrap_sa",
    "wrap_pt",
    "wrap_pt_tuned",
    "wrap_pt_houdayer",
    "wrap_pa",
    "wrap_pa_icm",
    "wrap_dimod",
    "sk_energy_density",
    "sk_parisi_reference_energy",
    "PARISI_SK_ENERGY_DENSITY",
    "pt_beta_ladder",
    "parallel_tempering_with_diagnostics",
    "parallel_tempering_betas",
    "auto_tune_beta_ladder",
    "auto_tune_beta_ladder_ktht",
    "wrap_pt_ktht",
    "records_to_csv",
    "records_to_json",
    "records_from_json",
]
