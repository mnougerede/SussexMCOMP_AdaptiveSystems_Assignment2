import numpy as np
import pytest
from types import SimpleNamespace

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from plasticity.hp import HP


def make_agent(z_vals, W=None, b=None):
    n = len(z_vals)
    agent = SimpleNamespace()
    agent.z = np.array(z_vals, dtype=float)
    agent.W = np.ones((n, n), dtype=float) if W is None else np.array(W, dtype=float)
    agent.b = np.zeros(n, dtype=float) if b is None else np.array(b, dtype=float)
    return agent


class TestHP:
    def setup_method(self):
        self.hp = HP()

    def test_low_activity_positive_rho(self):
        agent = make_agent([0.1])
        W_before = agent.W.copy()
        b_before = agent.b.copy()
        self.hp.step(agent)
        # rho = (0.2 - 0.1) / 0.2 = 0.5 > 0
        assert agent.b[0] > b_before[0]
        assert np.all(agent.W[0, :] > W_before[0, :])

    def test_mid_activity_zero_rho(self):
        agent = make_agent([0.5])
        W_before = agent.W.copy()
        b_before = agent.b.copy()
        self.hp.step(agent)
        np.testing.assert_array_equal(agent.b, b_before)
        np.testing.assert_array_equal(agent.W, W_before)

    def test_high_activity_negative_rho_positive_weight(self):
        # rho = (0.8 - 0.9) / (1 - 0.8) = -0.5 < 0
        W = [[2.0]]
        agent = make_agent([0.9], W=W)
        W_before = agent.W.copy()
        b_before = agent.b.copy()
        self.hp.step(agent)
        assert agent.b[0] < b_before[0]
        # positive W, rho < 0 → ΔW < 0 → |W| shrinks
        assert agent.W[0, 0] < W_before[0, 0]

    def test_high_activity_negative_rho_negative_weight(self):
        # negative weight: ΔW = rho * |W| < 0, so W becomes more negative → |W| grows
        W = [[-2.0]]
        agent = make_agent([0.9], W=W)
        W_before = agent.W.copy()
        self.hp.step(agent)
        assert agent.W[0, 0] < W_before[0, 0]
        assert np.abs(agent.W[0, 0]) > np.abs(W_before[0, 0])

    def test_boundary_h_l_zero_rho(self):
        agent = make_agent([0.2])  # z == H_L exactly
        W_before = agent.W.copy()
        b_before = agent.b.copy()
        self.hp.step(agent)
        np.testing.assert_array_equal(agent.b, b_before)
        np.testing.assert_array_equal(agent.W, W_before)

    def test_boundary_h_u_zero_rho(self):
        agent = make_agent([0.8])  # z == H_U exactly
        W_before = agent.W.copy()
        b_before = agent.b.copy()
        self.hp.step(agent)
        np.testing.assert_array_equal(agent.b, b_before)
        np.testing.assert_array_equal(agent.W, W_before)

    def test_disabled_is_noop(self):
        hp = HP(enabled=False)
        agent = make_agent([0.1, 0.9])
        W_before = agent.W.copy()
        b_before = agent.b.copy()
        hp.step(agent)
        np.testing.assert_array_equal(agent.W, W_before)
        np.testing.assert_array_equal(agent.b, b_before)

    def test_mixed_population(self):
        # Neuron 0: z=0.1 (< H_L), rho>0 → b↑, |W| row 0 ↑
        # Neuron 1: z=0.5 (in [H_L,H_U]), rho=0 → no change
        # Neuron 2: z=0.9 (> H_U), rho<0 → b↓, |W| row 2 shrinks for positive weights
        z = [0.1, 0.5, 0.9]
        W_init = np.ones((3, 3), dtype=float)
        agent = make_agent(z, W=W_init.copy())
        b_before = agent.b.copy()
        W_before = agent.W.copy()
        self.hp.step(agent)

        # Neuron 0 (low): b increases, all weights in row 0 increase
        assert agent.b[0] > b_before[0]
        assert np.all(agent.W[0, :] > W_before[0, :])

        # Neuron 1 (mid): no change
        assert agent.b[1] == b_before[1]
        np.testing.assert_array_equal(agent.W[1, :], W_before[1, :])

        # Neuron 2 (high): b decreases, positive weights in row 2 shrink
        assert agent.b[2] < b_before[2]
        assert np.all(agent.W[2, :] < W_before[2, :])

    def test_rho_values_match_formula(self):
        hp = HP(H_L=0.2, H_U=0.8, tau_w=40, tau_b=20, dt=0.2)
        z = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
        expected_rho = np.array([0.5, 0.0, 0.0, 0.0, -0.5])
        W = np.eye(5)
        agent = make_agent(z, W=W.copy(), b=np.zeros(5))
        b_before = agent.b.copy()
        hp.step(agent)
        actual_delta_b = agent.b - b_before
        np.testing.assert_allclose(
            actual_delta_b, (0.2 / 20) * expected_rho, rtol=1e-10
        )
