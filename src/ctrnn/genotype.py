import numpy as np

from ctrnn.agent import CTRNNAgent
from ctrnn.config import CTRNNConfig

WEIGHT_RANGE = (-10.0, 10.0)   # Williams 2006 §7.4.1.1
BIAS_RANGE   = (-10.0, 10.0)   # Williams 2006 §7.4.1.1
TAU_RANGE    = ( 1.0,   4.0)   # Williams 2006 §7.4.1.1


def _scale(g: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linearly map allele values from [-1, 1] to [lo, hi]."""
    return lo + (g + 1.0) * (hi - lo) / 2.0


def genotype_to_phenotype(g: np.ndarray, config: CTRNNConfig) -> dict[str, np.ndarray]:
    """Map a flat genotype vector to CTRNN parameter arrays.

    Layout (N = config.n_nodes):
      positions  0 .. N²-1  — weight matrix, row-major → shape (N, N)
      positions  N² .. N²+N-1 — biases → shape (N,)
      positions  N²+N .. N²+2N-1 — time constants → shape (N,)

    Allele values are assumed to lie in [-1, 1]; no validation is performed
    here — that is the GA's responsibility.

    Raises ValueError if g has the wrong length.
    """
    N = config.n_nodes
    expected = N * N + 2 * N
    if g.shape != (expected,):
        raise ValueError(f"Expected genotype shape ({expected},), got {g.shape}")

    weights = _scale(g[: N * N], *WEIGHT_RANGE).reshape(N, N)
    biases  = _scale(g[N * N : N * N + N], *BIAS_RANGE)
    taus    = _scale(g[N * N + N :],       *TAU_RANGE)

    return {"weights": weights, "biases": biases, "taus": taus}


def apply_genotype(g: np.ndarray, agent: CTRNNAgent) -> None:
    """Write the phenotype encoded by g directly onto agent's parameters."""
    p = genotype_to_phenotype(g, agent.config)
    agent.weights = p["weights"]
    agent.biases  = p["biases"]
    agent.taus    = p["taus"]
