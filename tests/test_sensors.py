import numpy as np
import pytest
from types import SimpleNamespace

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.sensors import compute_sensor_inputs


def make_shape(x, y, radius=10):
    s = SimpleNamespace()
    s.x = x
    s.y = y
    s.radius = radius
    return s


class TestSensors:
    def test_shape_directly_above_centre_sensor(self):
        # Agent (0,0), shape at (0,50), radius=10, centre ray θ=0
        # t_proj=50, d_perp=0, disc=100, t1=40, S=5*(100-40)/100=3.0
        shape = make_shape(0, 50, radius=10)
        sensors = compute_sensor_inputs(0, 0, shape)
        assert sensors[1] == pytest.approx(3.0)

    def test_shape_far_away_all_zero(self):
        # t1 = 200 - 10 = 190 > D_max=100 on centre ray → 0; side rays also miss
        shape = make_shape(0, 200, radius=10)
        sensors = compute_sensor_inputs(0, 0, shape)
        np.testing.assert_array_equal(sensors, [0.0, 0.0, 0.0])

    def test_shape_beyond_fan_all_zero(self):
        # Shape at (50, 10): angle from vertical ≈ 79° >> π/12 ≈ 15°
        # d_perp_sq ≈ 2089 >> radius²=100 for every ray
        shape = make_shape(50, 10, radius=10)
        sensors = compute_sensor_inputs(0, 0, shape)
        np.testing.assert_array_equal(sensors, [0.0, 0.0, 0.0])

    def test_all_three_sensors_hand_calculated(self):
        # Agent (0,0), shape at (0,50), radius=20, n_rays=3, fan_angle=π/6
        #
        # Centre ray (θ=0): direction=(0,1)
        #   t_proj=50, d_perp=0, disc=400, t1=30 → S=5*(100-30)/100=3.5
        #
        # Side rays (θ=±π/12): direction=(±sin(π/12), cos(π/12))
        #   t_proj = 50*cos(π/12)
        #   d_perp_sq = 2500*sin²(π/12)
        #   disc = 400 - 2500*sin²(π/12)
        #   t1 = 50*cos(π/12) - sqrt(disc)
        #   S = 5*(100-t1)/100
        theta = np.pi / 12
        t_proj_side = 50.0 * np.cos(theta)
        disc_side = 400.0 - 2500.0 * np.sin(theta) ** 2
        t1_side = t_proj_side - np.sqrt(disc_side)
        S_side = 5.0 * (100.0 - t1_side) / 100.0

        shape = make_shape(0, 50, radius=20)
        sensors = compute_sensor_inputs(0, 0, shape)
        np.testing.assert_allclose(sensors, [S_side, 3.5, S_side], rtol=1e-10)
