from dataclasses import dataclass


@dataclass
class EnvConfig:
    shape_radius: float = 10.0
    agent_radius: float = 5.0
    n_shapes: int = 20
    n_trials: int = 10
    spawn_height: float = 100.0
    spawn_x_range: float = 25.0
    vx_low: float = -0.3
    vx_high: float = 0.3
    vy_low: float = -0.5
    vy_high: float = -0.2
