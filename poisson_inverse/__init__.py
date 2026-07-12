"""Graph-Sobolev regularization for discrete Poisson inverse problems on DAGs."""
from .core import (
    empirical_flow, dominance_filter, hhd_dag_extract, top_k_prune,
    divergence, solve_poisson, poisson_residual,
    substochastic_operator, nilpotency_index,
)
from .synthetic import generate_funnel, build_chain, SyntheticFunnel
from .loaders import (
    load_retailrocket, load_trivago, load_otto, RealDataset,
)

__all__ = [
    "empirical_flow", "dominance_filter", "hhd_dag_extract", "top_k_prune",
    "divergence", "solve_poisson", "poisson_residual",
    "substochastic_operator", "nilpotency_index",
    "generate_funnel", "build_chain", "SyntheticFunnel",
    "load_retailrocket", "load_trivago", "load_otto", "RealDataset",
]
