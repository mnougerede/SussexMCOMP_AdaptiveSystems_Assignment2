import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.body import AgentBody


class TestAgentBody:
    def test_equal_inputs_no_movement(self):
        body = AgentBody(x=0.0)
        body.step(z_left=0.5, z_right=0.5, dt=0.2)
        assert body.x == 0.0

    def test_z_right_greater_moves_right(self):
        body = AgentBody(x=0.0)
        body.step(z_left=0.2, z_right=0.8, dt=0.2)
        assert body.x > 0.0

    def test_z_left_greater_moves_left(self):
        body = AgentBody(x=0.0)
        body.step(z_left=0.8, z_right=0.2, dt=0.2)
        assert body.x < 0.0

    def test_known_values(self):
        # dx = (0.2/0.2) * (0.8 - 0.2) = 1.0 * 0.6 = 0.6
        body = AgentBody(x=0.0, tau_x=0.2)
        body.step(z_left=0.2, z_right=0.8, dt=0.2)
        assert body.x == pytest.approx(0.6)
