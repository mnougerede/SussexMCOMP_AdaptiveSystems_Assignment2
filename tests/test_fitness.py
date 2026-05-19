import numpy as np
import pytest
from unittest.mock import patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environment.trial import TrialRecord
from environment.fitness import evaluate_fitness


def _make_record(shape_inits, body_xs_final, shape_xs_final):
    """Build a minimal TrialRecord from per-shape final positions."""
    n = len(shape_inits)
    body_xs = [np.array([bx]) for bx in body_xs_final]
    shape_xs = [np.array([sx]) for sx in shape_xs_final]
    shape_ys = [np.array([0.0]) for _ in range(n)]
    neural_states = [np.zeros((1, 5)) for _ in range(n)]
    return TrialRecord(body_xs, shape_xs, shape_ys, neural_states, shape_inits)


N = 20
VX, VY = 0.1, -0.5


def _inits(agent_x0, shape_x0):
    return [(shape_x0, VX, VY, agent_x0)] * N


class TestPerfectAgent:
    def test_fitness_approaches_one(self):
        # Agent x0 != shape x0 (S0 > 0), agent perfectly tracks shape (Sf = 0).
        # Each shape: phi(1 - 0/S0)=1, (1 - 0/Smax)=1 → contribution = 1.0
        agent_x0, shape_x0 = 0.0, 5.0
        record = _make_record(
            _inits(agent_x0, shape_x0),
            body_xs_final=[shape_x0] * N,   # agent ends at shape position
            shape_xs_final=[shape_x0] * N,
        )
        with patch('environment.fitness.run_trial', return_value=record):
            f = evaluate_fitness(None, None, None, n_trials=1)
        assert f == pytest.approx(1.0)


class TestSzeroGraceful:
    def test_s0_zero_no_exception(self):
        # Agent spawns directly under shape (S0 = 0). First term should be 0, not NaN/error.
        agent_x0 = shape_x0 = 10.0
        shape_x_final = shape_x0 + abs(VX) * (100.0 / abs(VY))
        record = _make_record(
            _inits(agent_x0, shape_x0),
            body_xs_final=[agent_x0] * N,       # agent stationary
            shape_xs_final=[shape_x_final] * N,
        )
        with patch('environment.fitness.run_trial', return_value=record):
            f = evaluate_fitness(None, None, None, n_trials=1)
        assert 0.0 <= f <= 1.0

    def test_s0_zero_first_term_is_zero(self):
        # With S0=0 and Sf=0 (agent stays at exact spawn point where shape also ends),
        # both terms collapse: first=0, second=(1 - 0/Smax)=1 → contribution=0.5.
        agent_x0 = shape_x0 = 0.0
        record = _make_record(
            _inits(agent_x0, shape_x0),
            body_xs_final=[0.0] * N,
            shape_xs_final=[0.0] * N,
        )
        with patch('environment.fitness.run_trial', return_value=record):
            f = evaluate_fitness(None, None, None, n_trials=1)
        assert f == pytest.approx(0.5)


class TestStationaryAgentFarFromShape:
    def test_fitness_low_but_nonnegative(self):
        # Agent starts at 0, shape starts at 25 (S0=25).
        # Agent stays at 0; shape drifts further away.
        agent_x0, shape_x0 = 0.0, 25.0
        shape_x_final = shape_x0 + abs(VX) * (100.0 / abs(VY))
        record = _make_record(
            _inits(agent_x0, shape_x0),
            body_xs_final=[agent_x0] * N,
            shape_xs_final=[shape_x_final] * N,
        )
        with patch('environment.fitness.run_trial', return_value=record):
            f = evaluate_fitness(None, None, None, n_trials=1)
        assert 0.0 <= f < 0.5


class TestFitnessInRange:
    def test_all_values_in_unit_interval(self):
        # Use a real agent so we exercise the full pipeline with varied outcomes.
        from ctrnn.config import CTRNNConfig
        from ctrnn.agent import CTRNNAgent
        from plasticity.hp import HP

        rng = np.random.default_rng(99)
        config = CTRNNConfig()
        agent = CTRNNAgent(config)
        agent.load_genotype(rng.uniform(-1, 1, config.genotype_length))
        hp = HP()

        f = evaluate_fitness(agent, hp, rng, n_trials=3)
        assert 0.0 <= f <= 1.0
