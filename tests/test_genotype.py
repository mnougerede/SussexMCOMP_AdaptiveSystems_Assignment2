import numpy as np
import pytest

from ctrnn.agent import CTRNNAgent
from ctrnn.config import CTRNNConfig
from ctrnn.genotype import apply_genotype, genotype_to_phenotype

N = 5
CFG = CTRNNConfig()


def test_shape_and_midpoint():
    """All-zero genotype maps to zero weights, zero biases, and taus at the
    midpoint of [1, 4] = 2.5."""
    p = genotype_to_phenotype(np.zeros(35), CFG)
    assert p["weights"].shape == (5, 5)
    assert p["biases"].shape == (5,)
    assert p["taus"].shape == (5,)
    np.testing.assert_array_equal(p["weights"], np.zeros((5, 5)))
    np.testing.assert_array_equal(p["biases"],  np.zeros(5))
    np.testing.assert_array_equal(p["taus"],    np.full(5, 2.5))


def test_endpoints_minus_one():
    """Genotype of all -1 maps to the lower bound of each range: weights=-10,
    biases=-10, taus=1."""
    p = genotype_to_phenotype(np.full(35, -1.0), CFG)
    np.testing.assert_array_equal(p["weights"], np.full((5, 5), -10.0))
    np.testing.assert_array_equal(p["biases"],  np.full(5, -10.0))
    np.testing.assert_array_equal(p["taus"],    np.full(5, 1.0))


def test_endpoints_plus_one():
    """Genotype of all +1 maps to the upper bound of each range: weights=+10,
    biases=+10, taus=4."""
    p = genotype_to_phenotype(np.full(35, 1.0), CFG)
    np.testing.assert_array_equal(p["weights"], np.full((5, 5), 10.0))
    np.testing.assert_array_equal(p["biases"],  np.full(5, 10.0))
    np.testing.assert_array_equal(p["taus"],    np.full(5, 4.0))


@pytest.mark.parametrize("k", [0, 1, N - 1, N, N * N - 1])
def test_weight_layout_row_major(k):
    """Weight alleles are laid out row-major: genotype index k -> matrix
    position (k // N, k % N). Checked at corners and edges of the matrix."""
    g = np.full(35, -1.0)
    g[k] = 1.0
    p = genotype_to_phenotype(g, CFG)

    expected = np.full((N, N), -10.0)
    expected[k // N, k % N] = 10.0
    np.testing.assert_array_equal(p["weights"], expected)


def test_bias_and_tau_positions():
    """Bias alleles start at index N², tau alleles at N²+N; a single +1 at each
    of these positions maps to the respective upper bound."""
    # single +1 at first bias position
    g = np.zeros(35)
    g[N * N] = 1.0
    p = genotype_to_phenotype(g, CFG)
    assert p["biases"][0] == 10.0
    np.testing.assert_array_equal(p["biases"][1:], np.zeros(N - 1))
    np.testing.assert_array_equal(p["weights"], np.zeros((N, N)))
    np.testing.assert_array_equal(p["taus"], np.full(N, 2.5))

    # single +1 at first tau position
    g = np.zeros(35)
    g[N * N + N] = 1.0
    p = genotype_to_phenotype(g, CFG)
    assert p["taus"][0] == 4.0
    np.testing.assert_array_equal(p["taus"][1:], np.full(N - 1, 2.5))


def test_wrong_shape_raises():
    """Genotype of wrong length raises ValueError."""
    with pytest.raises(ValueError):
        genotype_to_phenotype(np.zeros(34), CFG)


def test_apply_round_trip():
    """apply_genotype writes the same arrays that genotype_to_phenotype returns."""
    rng = np.random.default_rng(0)
    g = rng.uniform(-1.0, 1.0, size=35)

    expected = genotype_to_phenotype(g, CFG)
    agent = CTRNNAgent(CFG)
    apply_genotype(g, agent)

    np.testing.assert_allclose(agent.weights, expected["weights"])
    np.testing.assert_allclose(agent.biases,  expected["biases"])
    np.testing.assert_allclose(agent.taus,    expected["taus"])
