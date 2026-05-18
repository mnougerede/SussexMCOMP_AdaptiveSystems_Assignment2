class AgentBody:
    def __init__(self, x=0.0, radius=5.0, tau_x=0.2):
        self.x = x
        self.y = 0.0
        self.radius = radius
        self.tau_x = tau_x

    def step(self, z_left, z_right, dt):
        self.x += (dt / self.tau_x) * (z_right - z_left)
