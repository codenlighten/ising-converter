"""dimod interop: convert BQMs to our IsingModel and expose dimod-compatible samplers.

`dimod` is the D-Wave Ocean binary-quadratic-model interface. This adapter lets
ising_lab consume any BQM (Ising or QUBO) produced by Ocean tools, and exposes
our SA and PT kernels as `dimod.Sampler` subclasses so they can be plugged into
benchmark harnesses, hybrid workflows, or A/B comparisons against neal /
SimulatedAnnealingSampler / DWaveSampler / etc.

Importing this module requires the optional `dimod` dependency:
    pip install ising_lab[dimod]
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import dimod

from ._kernel import IsingModel, parallel_tempering, simulated_anneal
from .qubo import QUBO, qubo_to_ising


def from_bqm(bqm: "dimod.BinaryQuadraticModel") -> Tuple[IsingModel, float, List]:
    """Convert a dimod BQM (SPIN or BINARY) into (IsingModel, offset, labels).

    The relation `bqm.energy(sample) == ising_model.energy(spins) + offset` holds
    for any assignment, where `spins` is the sample's variables read off in the
    order given by `labels`. `labels` is a sorted list of the BQM's variables.
    """
    labels = sorted(bqm.variables)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)

    if bqm.vartype is dimod.SPIN:
        h = [float(bqm.linear.get(label, 0.0)) for label in labels]
        couplings = []
        for (u, v), w in bqm.quadratic.items():
            i, j = label_to_idx[u], label_to_idx[v]
            a, b = (i, j) if i < j else (j, i)
            couplings.append((a, b, float(w)))
        return IsingModel(n, h, couplings), float(bqm.offset), labels

    if bqm.vartype is dimod.BINARY:
        q = QUBO(num_vars=n, offset=float(bqm.offset))
        for label, w in bqm.linear.items():
            q.add_linear(label_to_idx[label], float(w))
        for (u, v), w in bqm.quadratic.items():
            q.add_quadratic(label_to_idx[u], label_to_idx[v], float(w))
        model, offset = qubo_to_ising(q)
        return model, offset, labels

    raise ValueError(f"unsupported vartype: {bqm.vartype}")


def to_bqm(
    model: IsingModel,
    offset: float = 0.0,
    labels: Sequence | None = None,
) -> "dimod.BinaryQuadraticModel":
    """Convert an IsingModel into a SPIN-valued dimod BQM.

    The returned BQM has `bqm.energy(spins) == model.energy(spins) + offset`.
    If `labels` is None, variables are integer indices 0..num_spins-1.
    """
    n = model.num_spins
    if labels is None:
        labels = list(range(n))
    elif len(labels) != n:
        raise ValueError(f"labels has length {len(labels)} but model has {n} spins")

    linear = {labels[i]: w for i, w in enumerate(model.h()) if w != 0.0}
    quadratic = {(labels[i], labels[j]): w for i, j, w in model.couplings()}
    return dimod.BinaryQuadraticModel(linear, quadratic, float(offset), dimod.SPIN)


def _build_sampleset(
    bqm: "dimod.BinaryQuadraticModel",
    raw: Sequence[Tuple[Sequence[int], float]],
    ising_offset: float,
    labels: Sequence,
) -> "dimod.SampleSet":
    """Wrap raw (spin_state, ising_energy) pairs into a dimod SampleSet."""
    binary = bqm.vartype is dimod.BINARY
    samples = []
    energies = []
    for state, ising_energy in raw:
        if binary:
            values = [(s + 1) // 2 for s in state]
        else:
            values = list(state)
        samples.append(dict(zip(labels, values)))
        energies.append(ising_energy + ising_offset)
    return dimod.SampleSet.from_samples(samples, vartype=bqm.vartype, energy=energies)


class SimulatedAnnealingSampler(dimod.Sampler):
    """dimod-compatible wrapper around `ising_lab.simulated_anneal`."""

    parameters = {
        "num_sweeps": [],
        "num_reads": [],
        "beta_start": [],
        "beta_end": [],
        "seed": [],
    }
    properties: dict = {}

    def sample(
        self,
        bqm: "dimod.BinaryQuadraticModel",
        num_sweeps: int = 1000,
        num_reads: int = 10,
        beta_start: float = 0.1,
        beta_end: float = 10.0,
        seed: int | None = None,
    ) -> "dimod.SampleSet":
        model, ising_offset, labels = from_bqm(bqm)
        raw = simulated_anneal(
            model,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            beta_start=beta_start,
            beta_end=beta_end,
            seed=seed,
        )
        return _build_sampleset(bqm, raw, ising_offset, labels)


class ParallelTemperingSampler(dimod.Sampler):
    """dimod-compatible wrapper around `ising_lab.parallel_tempering`.

    Each `num_reads` is one independent PT chain; the SampleSet records the
    best (state, energy) seen across all replicas during that chain.
    """

    parameters = {
        "num_sweeps": [],
        "num_replicas": [],
        "beta_min": [],
        "beta_max": [],
        "swap_every": [],
        "num_reads": [],
        "seed": [],
    }
    properties: dict = {}

    def sample(
        self,
        bqm: "dimod.BinaryQuadraticModel",
        num_sweeps: int = 1000,
        num_replicas: int = 8,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        swap_every: int = 1,
        num_reads: int = 10,
        seed: int | None = None,
    ) -> "dimod.SampleSet":
        model, ising_offset, labels = from_bqm(bqm)
        raw = parallel_tempering(
            model,
            num_sweeps=num_sweeps,
            num_replicas=num_replicas,
            beta_min=beta_min,
            beta_max=beta_max,
            swap_every=swap_every,
            num_reads=num_reads,
            seed=seed,
        )
        return _build_sampleset(bqm, raw, ising_offset, labels)


__all__ = [
    "from_bqm",
    "to_bqm",
    "SimulatedAnnealingSampler",
    "ParallelTemperingSampler",
]
