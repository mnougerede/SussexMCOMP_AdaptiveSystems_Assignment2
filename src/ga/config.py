from dataclasses import dataclass


@dataclass
class GAConfig:
    pop_size: int = 30
    n_gens: int = 300
    n_runs: int = 5
    n_elite: int = 5
    mutation_rate: float = 0.03
    mutation_add_magnitude: float = 0.5
