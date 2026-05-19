from dataclasses import dataclass


@dataclass
class GAConfig:
    pop_size: int = 30
    n_gens: int = 300
    n_runs: int = 5
    n_elite: int = 5
    p_m: float = 0.03
    sigma_m: float = 0.1
    tournament_k: int = 3
