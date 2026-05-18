import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ctrnn.config import CTRNNConfig
from ctrnn.agent import CTRNNAgent
from plasticity.hp import HP
from agent.body import AgentBody
from environment.trial import run_trial, TrialRecord


def make_agent(seed=0):
    config = CTRNNConfig()
    agent = CTRNNAgent(config)
    genotype = np.random.default_rng(seed).uniform(-1, 1, config.genotype_length)
    agent.load_genotype(genotype)
    return agent


class TestTrialDeterminism:
    def test_same_seed_identical_records(self):
        def run(agent_seed, rng_seed):
            agent = make_agent(agent_seed)
            hp = HP()
            body = AgentBody()
            rng = np.random.default_rng(rng_seed)
            return run_trial(agent, hp, body, rng, n_shapes=3)

        r1 = run(0, 7)
        r2 = run(0, 7)

        for a, b in zip(r1.body_xs, r2.body_xs):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(r1.neural_states, r2.neural_states):
            np.testing.assert_array_equal(a, b)
        assert r1.shape_inits == r2.shape_inits


class TestTrialStructure:
    def setup_method(self):
        self.agent = make_agent()
        self.hp = HP()
        self.body = AgentBody()
        self.rng = np.random.default_rng(42)

    def test_correct_number_of_shapes(self):
        record = run_trial(self.agent, self.hp, self.body, self.rng, n_shapes=5)
        assert len(record.body_xs) == 5

    def test_neural_states_shape(self):
        record = run_trial(self.agent, self.hp, self.body, self.rng, n_shapes=3)
        for ns in record.neural_states:
            assert ns.ndim == 2
            assert ns.shape[0] > 0
            assert ns.shape[1] == 5


class TestHPModeNone:
    def test_weights_unchanged_during_trial(self):
        agent = make_agent()
        W_before = agent.W.copy()
        b_before = agent.b.copy()
        hp = HP()
        body = AgentBody()
        rng = np.random.default_rng(1)
        run_trial(agent, hp, body, rng, n_shapes=2, hp_mode='none')
        np.testing.assert_array_equal(agent.W, W_before)
        np.testing.assert_array_equal(agent.b, b_before)


class TestHPModeDevelopment:
    def test_weights_differ_from_genotype_after_dev(self):
        agent = make_agent()
        W_genotype = agent.W.copy()
        hp = HP()
        body = AgentBody()
        rng = np.random.default_rng(2)
        run_trial(agent, hp, body, rng, n_shapes=2, hp_mode='development', dev_steps=200)
        assert not np.allclose(agent.W, W_genotype), "HP should have modified W during development"

    def test_hp_disabled_during_behaviour(self):
        # After a development trial, weights should equal what the dev phase alone produces.
        # If HP were active during behaviour they would diverge.
        config = CTRNNConfig()
        genotype = np.random.default_rng(0).uniform(-1, 1, config.genotype_length)
        dev_steps = 200

        # Full trial
        agent_trial = CTRNNAgent(config)
        agent_trial.load_genotype(genotype)
        hp_trial = HP()
        body_trial = AgentBody()
        run_trial(agent_trial, hp_trial, body_trial, np.random.default_rng(3),
                  n_shapes=3, hp_mode='development', dev_steps=dev_steps)
        W_trial = agent_trial.W.copy()
        b_trial = agent_trial.b.copy()

        # Manual dev-only phase with identical initial conditions
        agent_dev = CTRNNAgent(config)
        agent_dev.load_genotype(genotype)
        agent_dev.reset()
        hp_dev = HP()
        I_dev = np.zeros(config.n_nodes)
        for _ in range(dev_steps):
            agent_dev.step(I_dev)
            hp_dev.step(agent_dev)
        W_dev = agent_dev.W.copy()
        b_dev = agent_dev.b.copy()

        # Weights must match: behaviour phase did not alter them
        np.testing.assert_array_equal(W_trial, W_dev)
        np.testing.assert_array_equal(b_trial, b_dev)
