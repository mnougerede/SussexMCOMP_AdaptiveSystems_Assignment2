from dataclasses import dataclass


@dataclass
class CTRNNConfig:
    n_nodes: int = 5
    n_sensors: int = 3
    n_motors: int = 2
    dt: float = 0.2
    genotype_length: int = 35
