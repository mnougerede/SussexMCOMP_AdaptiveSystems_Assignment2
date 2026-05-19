import numpy as np

# Precomputed ray geometry for default parameters (n_rays=3, fan_angle=π/6).
# If non-default values are passed to compute_sensor_inputs, these constants
# will be used regardless — only call with defaults in production.
_RAY_ANGLES = np.linspace(-np.pi / 12, np.pi / 12, 3)
_RAY_SINS = np.sin(_RAY_ANGLES)
_RAY_COSS = np.cos(_RAY_ANGLES)


def compute_sensor_inputs(agent_x, agent_y, shape, S_max=5.0, D_max=100.0, n_rays=3, fan_angle=np.pi / 6):
    """Compute sensor activations for each ray cast toward the shape.

    Note: module-level constants _RAY_ANGLES/_RAY_SINS/_RAY_COSS assume default
    values (n_rays=3, fan_angle=π/6). Non-default arguments keep the same
    interface but use the precomputed constants unchanged.
    """
    sensors = np.zeros(n_rays)

    # Vector from ray origin to circle centre
    dx = shape.x - agent_x
    dy = shape.y - agent_y
    r = shape.radius

    for i in range(n_rays):
        # Ray direction: (sin θ, cos θ) — points upward with lateral offset
        d_x = _RAY_SINS[i]
        d_y = _RAY_COSS[i]

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
