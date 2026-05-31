"""ising_lab: convert combinatorial problems into Ising/QUBO form and solve them."""
from ._kernel import (
    IsingModel,
    belief_propagation,
    brute_force_ground_state,
    brute_force_min_energy,
    parallel_tempering,
    parallel_tempering_diagnostic,
    parallel_tempering_houdayer,
    population_annealing,
    population_annealing_icm,
    simulated_anneal,
)
from .qubo import QUBO, qubo_to_ising
from .registry import BestKnown, OptimumRegistry, sk_instance_key
from .inference import bp_ground_state, bp_marginals, loop_corrected_free_energy
from . import benchmarks, inference, problems

__all__ = [
    "IsingModel",
    "simulated_anneal",
    "parallel_tempering",
    "parallel_tempering_diagnostic",
    "parallel_tempering_houdayer",
    "population_annealing",
    "population_annealing_icm",
    "belief_propagation",
    "bp_marginals",
    "bp_ground_state",
    "loop_corrected_free_energy",
    "brute_force_min_energy",
    "brute_force_ground_state",
    "QUBO",
    "qubo_to_ising",
    "BestKnown",
    "OptimumRegistry",
    "sk_instance_key",
    "problems",
    "benchmarks",
    "inference",
]
