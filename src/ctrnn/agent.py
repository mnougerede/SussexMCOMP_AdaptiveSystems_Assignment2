import numpy as np

from ctrnn._madvn import CTRNN
from ctrnn.config import CTRNNConfig


class CTRNNAgent:
    """Thin adapter around madvn/CTRNN that constructs from CTRNNConfig and
    exposes the attribute surface expected by the HP and GA modules."""

    def __init__(self, config: CTRNNConfig):
        self.config = config
        self._net = CTRNN(size=config.n_nodes, step_size=config.dt)
        self._net.gains = np.ones(config.n_nodes)
        # PATCH: upstream initialises weights as scipy.sparse.csr_matrix; convert
        # to dense ndarray so HP (per-entry writes) and GA (whole-matrix writes)
        # work with plain numpy arrays throughout.
        self._net.weights = self._net.weights.toarray()

    # --- simulation -------------------------------------------------------

    def step(self, I: np.ndarray) -> None:
        """Advance one Euler step with external input vector I (length n_nodes)."""
        self._net.euler_step(I)

    def reset(self) -> None:
        """Zero all states; outputs become 0.5 via sigmoid(0)."""
        self._net.states = np.zeros(self.config.n_nodes)

    # --- parameter properties ---------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        # PATCH: guard in case _net.weights is ever set back to sparse externally.
        w = self._net.weights
        return w.toarray() if hasattr(w, "toarray") else np.asarray(w)

    @weights.setter
    def weights(self, value) -> None:
        # PATCH: enforce dense storage; HP and GA both expect a plain ndarray.
        self._net.weights = np.asarray(value)

    @property
    def biases(self):
        return self._net.biases

    @biases.setter
    def biases(self, value):
        self._net.biases = value

    @property
    def taus(self):
        return self._net.taus

    @taus.setter
    def taus(self, value):
        self._net.taus = value

    # --- state properties -------------------------------------------------

    @property
    def y(self):
        return self._net.states

    @y.setter
    def y(self, value):
        self._net.states = value

    @property
    def z(self):
        return self._net.outputs

    @z.setter
    def z(self, value):
        self._net.outputs = value

    # --- derived read-only ------------------------------------------------

    @property
    def motor_outputs(self) -> np.ndarray:
        """Firing rates of motor neurons: z[n_sensors : n_sensors + n_motors]."""
        s = self.config.n_sensors
        m = self.config.n_motors
        return self.z[s : s + m]
