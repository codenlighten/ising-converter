"""Canonical problem encoders: combinatorial problem -> Ising or QUBO.

These are reference implementations of the recipes in Lucas (2014),
'Ising formulations of many NP problems'.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from ._kernel import IsingModel
from .qubo import QUBO, qubo_to_ising, spins_to_bits


def max_cut(num_nodes: int, edges: Iterable[Tuple[int, int]]) -> Tuple[IsingModel, float]:
    """Max-Cut on an unweighted graph as an Ising minimization.

    H(s) = sum_{(i,j) in E} (s_i s_j - 1) / 2
    so that cut_size = -H. Returns (model, offset) where
        cut_size(state) = -(model.energy(state) + offset).
    """
    edges = list(edges)
    h = [0.0] * num_nodes
    couplings: list[Tuple[int, int, float]] = []
    offset = 0.0
    for i, j in edges:
        if i == j:
            raise ValueError("self-loops not allowed in max-cut")
        a, b = (i, j) if i < j else (j, i)
        couplings.append((a, b, 0.5))
        offset += -0.5
    return IsingModel(num_nodes, h, couplings), offset


def number_partition(numbers: Sequence[float]) -> Tuple[IsingModel, float]:
    """Number partitioning: split `numbers` into two groups of equal sum.

    Variables s_i in {-1, +1} assign each number to a group.
    H(s) = (sum_i n_i s_i)^2 = sum_i n_i^2 + 2 sum_{i<j} n_i n_j s_i s_j
    so h_i = 0 and J_ij = 2 n_i n_j. Offset is sum_i n_i^2.
    A ground state with H = 0 means a perfect partition exists.
    """
    n = len(numbers)
    h = [0.0] * n
    couplings: list[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            couplings.append((i, j, 2.0 * numbers[i] * numbers[j]))
    offset = sum(x * x for x in numbers)
    return IsingModel(n, h, couplings), offset


def cardinality_qubo(num_vars: int, k: int, penalty: float) -> QUBO:
    """Penalty QUBO enforcing sum_i x_i = k: A * (sum_i x_i - k)^2."""
    q = QUBO(num_vars=num_vars, offset=penalty * k * k)
    for i in range(num_vars):
        q.add_linear(i, penalty * (1 - 2 * k))
        for j in range(i + 1, num_vars):
            q.add_quadratic(i, j, 2.0 * penalty)
    return q


def tsp(
    distances: Sequence[Sequence[float]],
    penalty: float | None = None,
) -> Tuple[IsingModel, float]:
    """Traveling salesman as Ising. `distances[u][v]` is u->v cost (may be asymmetric).

    Variables x[v, j] in {0, 1} mean "city v is visited at tour position j",
    flattened to index v * N + j. The Lucas (2014) Hamiltonian is

        H = A * sum_v (sum_j x[v,j] - 1)^2     # each city visited once
          + A * sum_j (sum_v x[v,j] - 1)^2     # each position holds one city
          + sum_{j, u != v} d[u][v] x[u, j] x[v, (j+1) mod N]

    For the ground state to be a valid tour, A must dominate any cost savings
    from violating a constraint. We pick A = N * max|d| + 1 by default, which
    is safe for any feasible tour. Pass `penalty` to override.

    Decode with `tsp_decode_tour(state, num_cities)`.
    """
    n = len(distances)
    if n < 2:
        raise ValueError("TSP needs at least 2 cities")
    for row in distances:
        if len(row) != n:
            raise ValueError("distances must be square")

    if penalty is None:
        max_w = max((abs(distances[u][v]) for u in range(n) for v in range(n)), default=0.0)
        penalty = max_w * n + 1.0

    def idx(v: int, j: int) -> int:
        return v * n + j

    q = QUBO(num_vars=n * n)

    # (sum_j x[v,j] - 1)^2 = -sum_j x[v,j] + 2 sum_{j<k} x[v,j] x[v,k] + 1   (binary)
    for v in range(n):
        for j in range(n):
            q.add_linear(idx(v, j), -penalty)
            for k in range(j + 1, n):
                q.add_quadratic(idx(v, j), idx(v, k), 2.0 * penalty)
        q.offset += penalty

    for j in range(n):
        for v in range(n):
            q.add_linear(idx(v, j), -penalty)
            for u in range(v + 1, n):
                q.add_quadratic(idx(v, j), idx(u, j), 2.0 * penalty)
        q.offset += penalty

    for j in range(n):
        jn = (j + 1) % n
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                w = distances[u][v]
                if w != 0.0:
                    q.add_quadratic(idx(u, j), idx(v, jn), w)

    return qubo_to_ising(q)


def tsp_decode_tour(state: Sequence[int], num_cities: int) -> list[int]:
    """Decode a TSP spin state into a tour [city_at_pos_0, city_at_pos_1, ...].

    Raises ValueError if the assignment violates a row/column constraint.
    """
    n = num_cities
    bits = spins_to_bits(state)
    if len(bits) != n * n:
        raise ValueError(f"state has {len(bits)} entries, expected {n * n}")
    tour = [-1] * n
    used = [False] * n
    for j in range(n):
        chosen = [v for v in range(n) if bits[v * n + j] == 1]
        if len(chosen) != 1:
            raise ValueError(f"position {j} has {len(chosen)} cities (expected 1)")
        v = chosen[0]
        if used[v]:
            raise ValueError(f"city {v} appears at multiple positions")
        used[v] = True
        tour[j] = v
    return tour


def tour_cost(tour: Sequence[int], distances: Sequence[Sequence[float]]) -> float:
    """Total closed-loop cost of `tour` under `distances`."""
    n = len(tour)
    return sum(distances[tour[j]][tour[(j + 1) % n]] for j in range(n))


def graph_coloring(
    num_nodes: int,
    edges: Iterable[Tuple[int, int]],
    num_colors: int,
    penalty: float = 1.0,
) -> Tuple[IsingModel, float]:
    """K-coloring decision problem as an Ising minimization.

    Variables x[v, c] in {0, 1} mean "vertex v has color c", flattened to v * K + c.
    The QUBO is

        H = A * sum_v (sum_c x[v, c] - 1)^2          # exactly one color per vertex
          + A * sum_{(u, v) in E} sum_c x[u, c] x[v, c]   # no monochromatic edges

    With A > 0, ground-state energy = 0 iff the graph is K-colorable. The energy
    above 0 counts violations (uncolored/multi-colored vertices and monochromatic
    edges), weighted by `penalty`.

    Decode with `graph_coloring_decode(state, num_nodes, num_colors)`.
    """
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")
    if num_colors < 1:
        raise ValueError("num_colors must be >= 1")
    if penalty <= 0.0:
        raise ValueError("penalty must be > 0")

    k = num_colors

    def idx(v: int, c: int) -> int:
        return v * k + c

    q = QUBO(num_vars=num_nodes * k)

    # Row constraint: each vertex gets exactly one color.
    # (sum_c x[v,c] - 1)^2 expands (binary) to -sum_c x[v,c] + 2 sum_{c<d} x[v,c] x[v,d] + 1.
    for v in range(num_nodes):
        for c in range(k):
            q.add_linear(idx(v, c), -penalty)
            for d in range(c + 1, k):
                q.add_quadratic(idx(v, c), idx(v, d), 2.0 * penalty)
        q.offset += penalty

    # Edge constraint: penalize same-color endpoints on every edge.
    seen_edges = set()
    for u, v in edges:
        if u == v:
            raise ValueError("self-loops not allowed in graph coloring")
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"edge ({u}, {v}) references out-of-range vertex")
        key = (min(u, v), max(u, v))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        for c in range(k):
            q.add_quadratic(idx(u, c), idx(v, c), penalty)

    return qubo_to_ising(q)


def knapsack(
    weights: Sequence[int],
    values: Sequence[float],
    capacity: int,
    penalty: float | None = None,
) -> Tuple[IsingModel, float]:
    """0/1 knapsack: maximize total value subject to a weight ceiling.

    Variables:
        x[0 .. N-1]            -- item-inclusion bits
        y[N .. N + B - 1]      -- slack bits with weights 2^b
    where B = ceil(log2(capacity + 1)). The inequality
    sum_i w_i x_i <= capacity is converted to an equality
    sum_i w_i x_i + sum_b 2^b y_b = capacity, penalized quadratically:

        H = -sum_i v_i x_i + A * (sum_i w_i x_i + sum_b 2^b y_b - capacity)^2

    Weights and capacity must be non-negative integers (slack bits represent
    integers). The default penalty A = sum_i |v_i| + 1 dominates any value
    gain from violating the constraint.

    Decode with `knapsack_decode(state, num_items)`. The slack bits at the
    tail of the state can be discarded.
    """
    n = len(weights)
    if n != len(values):
        raise ValueError("weights and values must have the same length")
    if capacity < 0:
        raise ValueError("capacity must be >= 0")
    parsed_weights: list[int] = []
    for w in weights:
        iw = int(w)
        if iw != w or iw < 0:
            raise ValueError("weights must be non-negative integers")
        parsed_weights.append(iw)

    num_slack = capacity.bit_length()  # 0 when capacity == 0
    num_vars = n + num_slack

    if penalty is None:
        penalty = sum(abs(float(v)) for v in values) + 1.0
    if penalty <= 0.0:
        raise ValueError("penalty must be > 0")

    coeffs: list[float] = [float(w) for w in parsed_weights]
    coeffs.extend(float(1 << b) for b in range(num_slack))

    q = QUBO(num_vars=num_vars)

    # Objective: minimize -value(x).
    for i, v in enumerate(values):
        if v != 0.0:
            q.add_linear(i, -float(v))

    # Penalty: A * (sum_k c_k x_k - C)^2
    #        = A * sum_k c_k(c_k - 2C) x_k + 2A sum_{k<l} c_k c_l x_k x_l + A C^2.
    cap = float(capacity)
    for k, ck in enumerate(coeffs):
        q.add_linear(k, penalty * ck * (ck - 2.0 * cap))
        for l_idx in range(k + 1, num_vars):
            cl = coeffs[l_idx]
            q.add_quadratic(k, l_idx, 2.0 * penalty * ck * cl)
    q.offset += penalty * cap * cap

    return qubo_to_ising(q)


def knapsack_decode(state: Sequence[int], num_items: int) -> list[int]:
    """Return the sorted list of selected item indices from a knapsack spin state."""
    bits = spins_to_bits(state)
    if len(bits) < num_items:
        raise ValueError(f"state has {len(bits)} entries, fewer than num_items={num_items}")
    return [i for i in range(num_items) if bits[i] == 1]


def graph_coloring_decode(
    state: Sequence[int], num_nodes: int, num_colors: int
) -> list[int]:
    """Decode a spin state into a color per vertex.

    Returns a list `colors` with `colors[v]` in `[0, num_colors)`.
    Raises ValueError if any vertex has != 1 color bit set.
    """
    bits = spins_to_bits(state)
    if len(bits) != num_nodes * num_colors:
        raise ValueError(
            f"state has {len(bits)} entries, expected {num_nodes * num_colors}"
        )
    colors = [-1] * num_nodes
    for v in range(num_nodes):
        chosen = [c for c in range(num_colors) if bits[v * num_colors + c] == 1]
        if len(chosen) != 1:
            raise ValueError(f"vertex {v} has {len(chosen)} colors (expected 1)")
        colors[v] = chosen[0]
    return colors


__all__ = [
    "max_cut",
    "number_partition",
    "cardinality_qubo",
    "tsp",
    "tsp_decode_tour",
    "tour_cost",
    "graph_coloring",
    "graph_coloring_decode",
    "knapsack",
    "knapsack_decode",
    "qubo_to_ising",
]
