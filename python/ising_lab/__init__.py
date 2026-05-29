"""ising_lab: convert combinatorial problems into Ising/QUBO form and solve them."""
from ._kernel import (
    IsingModel,
    brute_force_ground_state,
    brute_force_min_energy,
    parallel_tempering,
    parallel_tempering_diagnostic,
    simulated_anneal,
)
from .qubo import QUBO, qubo_to_ising
from .registry import BestKnown, OptimumRegistry, sk_instance_key
from . import benchmarks, problems

__all__ = [
    "IsingModel",
    "simulated_anneal",
    "parallel_tempering",
    "parallel_tempering_diagnostic",
    "brute_force_min_energy",
    "brute_force_ground_state",
    "QUBO",
    "qubo_to_ising",
    "BestKnown",
    "OptimumRegistry",
    "sk_instance_key",
    "problems",
    "benchmarks",
]
