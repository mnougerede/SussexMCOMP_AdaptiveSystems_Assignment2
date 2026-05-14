import math
from dataclasses import dataclass


@dataclass
class AgentConfig:
    n_rays: int = 3
    ray_spread: float = math.pi / 6
    s_max: float = 5.0
    d_max: float = 100.0
    tau_x: float = 0.2
