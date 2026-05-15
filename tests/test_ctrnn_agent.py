import numpy as np
import pytest

from ctrnn.agent import CTRNNAgent
from ctrnn.config import CTRNNConfig


def make_agent():
    return CTRNNAgent(CTRNNConfig())


def null_network(agent):
    """Zero weights, zero biases, unit taus — isolates state dynamics."""
    agent.weights = np.zeros((agent.config.n_nodes, agent.config.n_nodes))
    agent.biases = np.zeros(agent.config.n_nodes)
    agent.taus = np.ones(agent.config.n_nodes)


def test_construction():
    """CTRNNAgent initialises with arrays of the correct shape for a default 5-node config."""
    agent = make_agent()
    assert agent.weights.shape == (5, 5)
    assert agent.biases.shape == (5,)
    assert agent.taus.shape == (5,)
    assert agent.y.shape == (5,)
    assert agent.z.shape == (5,)


def test_decay_to_zero_state():
    """With zero weights, zero biases, and unit taus, dy/dt = -y so y -> 0 and z -> sigmoid(0) = 0.5."""
    agent = make_agent()
    null_network(agent)
    agent.y = np.ones(5)

    I = np.zeros(5)
    for _ in range(200):
        agent.step(I)

    np.testing.assert_allclose(agent.y, np.zeros(5), atol=1e-3)
    np.testing.assert_allclose(agent.z, np.full(5, 0.5), atol=1e-3)


def test_constant_input_equilibrium():
    """With zero weights and unit taus, dy_i/dt = -y_i + I_i has fixed point y_i = I_i.
    Sensor nodes (0-2) receive non-zero input; motor nodes (3-4) receive zero, locking in
    the sensor-neurons-first indexing convention."""
    agent = make_agent()
    null_network(agent)

    I = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    for _ in range(200):
        agent.step(I)

    np.testing.assert_allclose(agent.y, I, atol=1e-3)


def test_motor_outputs_slice():
    """motor_outputs returns z[n_sensors : n_sensors + n_motors]; with extreme states
    the first three outputs saturate near 0 and the last two near 1."""
    agent = make_agent()
    agent.biases = np.zeros(5)
    agent.y = np.array([-100.0, -100.0, -100.0, 100.0, 100.0])

    np.testing.assert_allclose(agent.z[:3], np.zeros(3), atol=1e-3)
    np.testing.assert_allclose(agent.motor_outputs, np.ones(2), atol=1e-3)
    np.testing.assert_array_equal(agent.motor_outputs, agent.z[3:5])


def test_reset():
    """After reset(), y is exactly zero and z equals sigmoid(biases). With the
    default-constructed agent's zero biases, this reduces to z = 0.5."""
    agent = make_agent()
    I = np.ones(5)
    for _ in range(10):
        agent.step(I)

    agent.reset()

    np.testing.assert_allclose(agent.y, np.zeros(5), atol=1e-9)
    np.testing.assert_allclose(agent.z, np.full(5, 0.5), atol=1e-9)


def test_weights_is_dense_ndarray():
    """agent.weights must be a plain numpy ndarray, not a scipy sparse matrix.
    HP writes individual entries and GA writes whole matrices; both require dense."""
    agent = make_agent()
    assert isinstance(agent.weights, np.ndarray)
