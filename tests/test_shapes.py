import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environment.shapes import FallingShape, make_shape


class TestFallingShape:
    def test_step_updates_position(self):
        shape = FallingShape(x=0.0, y=100.0, vx=1.0, vy=-0.5)
        shape.step(2.0)
        assert shape.x == pytest.approx(2.0)
        assert shape.y == pytest.approx(99.0)

    def test_has_passed_false_when_above(self):
        # bottom of shape = 50 - 10 = 40; agent top = 0 + 10 = 10; 40 > 10 → not passed
        shape = FallingShape(x=0.0, y=50.0, vx=0.0, vy=-0.3, radius=10)
        assert shape.has_passed(agent_y=0, agent_radius=10) is False

    def test_has_passed_true_when_below(self):
        # bottom of shape = 5 - 10 = -5; agent top = 0 + 10 = 10; -5 < 10 → passed
        shape = FallingShape(x=0.0, y=5.0, vx=0.0, vy=-0.3, radius=10)
        assert shape.has_passed(agent_y=0, agent_radius=10) is True


class TestMakeShape:
    def setup_method(self):
        self.rng = np.random.default_rng(42)

    def test_x_in_range(self):
        agent_x = 50.0
        for _ in range(200):
            shape = make_shape(agent_x, self.rng)
            assert agent_x - 25 <= shape.x <= agent_x + 25

    def test_vy_in_range(self):
        for _ in range(200):
            shape = make_shape(0.0, self.rng)
            assert -0.5 <= shape.vy <= -0.2

    def test_vx_in_range(self):
        for _ in range(200):
            shape = make_shape(0.0, self.rng)
            assert -0.3 <= shape.vx <= 0.3
