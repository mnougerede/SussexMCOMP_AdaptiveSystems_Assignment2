import numpy as np


def compute_sensor_inputs(agent_x, agent_y, shape, S_max=5.0, D_max=100.0, n_rays=3, fan_angle=np.pi / 6):
    half = fan_angle / 2
    angles = np.linspace(-half, half, n_rays)

    sensors = np.zeros(n_rays)

    # Vector from ray origin to circle centre
    dx = shape.x - agent_x
    dy = shape.y - agent_y
    r = shape.radius

    for i, theta in enumerate(angles):
        # Ray direction: (sin θ, cos θ) — points upward with lateral offset
        d_x = np.sin(theta)
        d_y = np.cos(theta)

        t_proj = dx * d_x + dy * d_y
        d_perp_sq = dx ** 2 + dy ** 2 - t_proj ** 2

        discriminant = r ** 2 - d_perp_sq
        if discriminant < 0:
            continue

        sqrt_disc = np.sqrt(discriminant)
        t1 = t_proj - sqrt_disc
        t2 = t_proj + sqrt_disc

        if t1 > 0:
            D = t1
        elif t2 > 0:
            D = t2
        else:
            continue

        if D > D_max:
            continue

        sensors[i] = S_max * (D_max - D) / D_max

    return sensors
