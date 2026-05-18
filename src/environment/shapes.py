import numpy as np


class FallingShape:
    def __init__(self, x, y, vx, vy, radius=10):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius

    def step(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def has_passed(self, agent_y, agent_radius):
        return self.y - self.radius < agent_y + agent_radius


def make_shape(agent_x, rng):
    agent_y = 0
    x = agent_x + rng.uniform(-25, 25)
    y = agent_y + 100
    vx = rng.uniform(-0.3, 0.3)
    vy = rng.uniform(-0.5, -0.2)
    return FallingShape(x, y, vx, vy)
