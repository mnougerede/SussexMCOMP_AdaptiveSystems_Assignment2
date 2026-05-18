import numpy as np


class HP:
    def __init__(self, H_L=0.2, H_U=0.8, tau_w=40, tau_b=20, dt=0.2, enabled=True):
        self.H_L = H_L
        self.H_U = H_U
        self.tau_w = tau_w
        self.tau_b = tau_b
        self.dt = dt
        self.enabled = enabled

    def step(self, agent):
        if not self.enabled:
            return

        z = agent.z
        rho = np.where(
            z < self.H_L,
            (self.H_L - z) / self.H_L,
            np.where(z > self.H_U, (self.H_U - z) / (1.0 - self.H_U), 0.0),
        )

        agent.b += (self.dt / self.tau_b) * rho
        agent.W += (self.dt / self.tau_w) * rho[:, np.newaxis] * np.abs(agent.W)
