"""QUBO model and conversion to Ising form.

QUBO:  f(x) = offset + sum_i Q_i x_i + sum_{i<j} Q_ij x_i x_j,   x_i in {0, 1}
Ising: H(s) = offset + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j,   s_i in {-1, +1}

Substitution x_i = (1 + s_i) / 2 gives the conversion below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

from ._kernel import IsingModel


@dataclass
class QUBO:
    num_vars: int
    linear: Dict[int, float] = field(default_factory=dict)
    quadratic: Dict[Tuple[int, int], float] = field(default_factory=dict)
    offset: float = 0.0

    def add_linear(self, i: int, w: float) -> None:
        self._check_index(i)
        self.linear[i] = self.linear.get(i, 0.0) + w

    def add_quadratic(self, i: int, j: int, w: float) -> None:
        if i == j:
            self.add_linear(i, w)
            return
        self._check_index(i)
        self._check_index(j)
        a, b = (i, j) if i < j else (j, i)
        self.quadratic[(a, b)] = self.quadratic.get((a, b), 0.0) + w

    def energy(self, x: Sequence[int]) -> float:
        e = self.offset
        for i, w in self.linear.items():
            e += w * x[i]
        for (i, j), w in self.quadratic.items():
            e += w * x[i] * x[j]
        return e

    def _check_index(self, i: int) -> None:
        if not 0 <= i < self.num_vars:
            raise IndexError(f"variable {i} out of range [0, {self.num_vars})")


def qubo_to_ising(qubo: QUBO) -> Tuple[IsingModel, float]:
    """Convert a QUBO to (IsingModel, offset). Ising energy + offset == QUBO energy."""
    n = qubo.num_vars
    h = [0.0] * n
    j_map: Dict[Tuple[int, int], float] = {}
    offset = qubo.offset

    # Linear: Q_i * x_i = (Q_i / 2) * s_i + Q_i / 2
    for i, w in qubo.linear.items():
        h[i] += w / 2.0
        offset += w / 2.0

    # Quadratic: Q_ij * x_i * x_j = (Q_ij / 4) * (1 + s_i + s_j + s_i * s_j)
    for (i, j), w in qubo.quadratic.items():
        a, b = (i, j) if i < j else (j, i)
        h[a] += w / 4.0
        h[b] += w / 4.0
        j_map[(a, b)] = j_map.get((a, b), 0.0) + w / 4.0
        offset += w / 4.0

    couplings = [(a, b, w) for (a, b), w in j_map.items() if w != 0.0]
    return IsingModel(n, h, couplings), offset


def spins_to_bits(state: Sequence[int]) -> list[int]:
    """Map an Ising state in {-1, +1} to a QUBO assignment in {0, 1}."""
    return [(s + 1) // 2 for s in state]
