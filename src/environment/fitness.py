import numpy as np

from agent.body import AgentBody
from environment.trial import run_trial


def _phi(x):
    return x if 0.0 <= x <= 1.0 else 0.0


def _shape_contribution(shape_init, body_xs_i, shape_xs_i):
    shape_x0, vx, vy, agent_x0 = shape_init
    S0 = abs(agent_x0 - shape_x0)
    Sf = abs(body_xs_i[-1] - shape_xs_i[-1])
    Smax = (1.0 + abs(vx)) * (100.0 / abs(vy))

    first = 0.0 if S0 == 0.0 else _phi(1.0 - Sf / S0)
    second = max(0.0, min(1.0, 1.0 - Sf / Smax))
    return 0.5 * (first + second)


def evaluate_fitness(agent, hp, rng, n_trials=10, n_shapes=20, hp_mode='none'):
    total = 0.0
    for _ in range(n_trials):
        body = AgentBody()
        record = run_trial(agent, hp, body, rng, n_shapes=n_shapes, hp_mode=hp_mode)
        n_shapes = len(record.shape_inits)
        trial_sum = sum(
            _shape_contribution(record.shape_inits[i], record.body_xs[i], record.shape_xs[i])
            for i in range(n_shapes)
        )
        total += trial_sum / n_shapes
    return total / n_trials
