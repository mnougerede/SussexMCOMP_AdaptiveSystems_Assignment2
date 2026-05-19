import numpy as np
from dataclasses import dataclass

from environment.shapes import make_shape
from agent.sensors import compute_sensor_inputs


@dataclass
class TrialRecord:
    body_xs: list
    shape_xs: list
    shape_ys: list
    neural_states: list
    shape_inits: list


def run_trial(agent, hp, body, rng, n_shapes=20, hp_mode='none', dev_steps=6000):
    dt = 0.2

    agent.load_genotype(agent.genotype)
    agent.reset()
    body.x = 0.0

    hp.enabled = False

    if hp_mode in ('development', 'both'):
        hp.enabled = True
        I_dev = np.zeros(agent.config.n_nodes)
        for _ in range(dev_steps):
            agent.step(I_dev)
            hp.step(agent)
        if hp_mode != 'both':
            hp.enabled = False

    if hp_mode in ('behaviour', 'both'):
        hp.enabled = True

    body_xs, shape_xs, shape_ys, neural_states, shape_inits = [], [], [], [], []

    for _ in range(n_shapes):
        shape = make_shape(body.x, rng)
        shape_inits.append((shape.x, shape.vx, shape.vy, body.x))

        bx_t, sx_t, sy_t, zs_t = [], [], [], []

        while not shape.has_passed(body.y, body.radius):
            I = np.zeros(agent.config.n_nodes)
            I[:agent.config.n_sensors] = compute_sensor_inputs(
                body.x, body.y, shape, n_rays=agent.config.n_sensors
            )
            agent.step(I)
            if hp.enabled:
                hp.step(agent)
            body.step(agent.z[3], agent.z[4], dt)
            shape.step(dt)

            bx_t.append(body.x)
            sx_t.append(shape.x)
            sy_t.append(shape.y)
            zs_t.append(agent.z.copy())

        body_xs.append(np.array(bx_t))
        shape_xs.append(np.array(sx_t))
        shape_ys.append(np.array(sy_t))
        neural_states.append(np.array(zs_t))

    return TrialRecord(
        body_xs=body_xs,
        shape_xs=shape_xs,
        shape_ys=shape_ys,
        neural_states=neural_states,
        shape_inits=shape_inits,
    )
