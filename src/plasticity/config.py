from dataclasses import dataclass


@dataclass
class HPConfig:
    h_low: float = 0.2
    h_high: float = 0.8
    tau_w: float = 40.0
    tau_b: float = 20.0
