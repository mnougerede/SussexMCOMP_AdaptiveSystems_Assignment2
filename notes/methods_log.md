# Methods log

A working document that accrues methods-section content as the project is built. Each section starts as bulleted decisions + citations and fills in as the corresponding code is written. By the end of Phase 5, this should read as first-draft methods prose.

Conventions:
- Citations are inline as `(Williams 2006, §7.4.1.1)` style; full references compiled at the end.
- Williams quotes use the **Chapter 7** values throughout. Where Chapter 6 / conference paper values differ, this is flagged explicitly.
- Every variable in every equation is defined on the same logical block, even if redundantly.
- `[TBD: Pass N]` markers indicate content that will be written when that pass is built.

---

## 1. System overview

The experimental system has four coupled components:

1. **Falling-shape environment.** Circles of fixed radius enter the simulation above the agent at randomised horizontal offsets and velocities, and fall through the agent's sensor fan until they pass below it.
2. **Agent body.** A circular body constrained to horizontal motion, equipped with a fan of upward-pointing ray sensors.
3. **CTRNN controller.** A 5-node fully-connected continuous-time recurrent neural network; three nodes receive ray sensor input as external current, two nodes drive the agent's left/right motors.
4. **Homeostatic plasticity (HP) module.** A separate adaptive process operating on the CTRNN's weights and biases, parameterised independently of the network's dynamics.

A genetic algorithm operates over the CTRNN's weights, biases and per-neuron time constants. HP is **not** evolved — its parameters are fixed across all runs. The experimental conditions vary *when* HP is applied (during a developmental phase before each fitness trial, during the trial itself, both, or never), not *whether* it acts.

System diagram: `[TBD: notes/system_diagram.png — Phase 1 deliverable]`.

---

## 2. CTRNN model

### 2.1 State equation

The CTRNN comprises $N = 5$ fully-connected nodes. Each node $i$ has an internal state $y_i$ (real-valued, unbounded) and a firing rate $z_i \in (0, 1)$. The state evolves according to (Beer 1995):

$$
\tau_{y,i} \dot{y}_i \;=\; -y_i \;+\; \sum_{j=1}^{N} w_{ji} z_j \;+\; I_i
\qquad (1)
$$

where:
- $y_i$ is the membrane state of node $i$;
- $\tau_{y,i} > 0$ is the membrane time constant of node $i$ (evolved per node; see §2.4);
- $w_{ji}$ is the synaptic weight from presynaptic node $j$ to postsynaptic node $i$;
- $z_j$ is the firing rate of presynaptic node $j$ (defined in eq. 2);
- $I_i$ is the external input to node $i$, used to inject sensor signal (see §4).

### 2.2 Activation function

The firing rate is the standard logistic sigmoid of the biased state:

$$
z_i \;=\; \sigma(y_i + b_i) \;=\; \frac{1}{1 + e^{-(y_i + b_i)}}
\qquad (2)
$$

where $b_i \in \mathbb{R}$ is the bias of node $i$, evolved (and, under HP, also adjusted online).

Note: gain is implicitly 1. `madvn/CTRNN` (see §2.5) supports a per-neuron gain parameter; we set this to 1 throughout to match Williams' Chapter 7 specification, which does not evolve gains.

### 2.3 Integration

Equation (1) is integrated using Euler's forward method with step size $\Delta t = 0.2$ (Williams 2006, §7.4.1.1):

$$
y_i(t + \Delta t) \;=\; y_i(t) \;+\; \frac{\Delta t}{\tau_{y,i}} \Big( -y_i(t) + \sum_j w_{ji} z_j(t) + I_i(t) \Big)
\qquad (3)
$$

### 2.4 Parameter ranges

The genotype encodes $N^2 + 2N = 35$ real-valued alleles, each in $[-1, 1]$, mapped linearly to the phenotypic ranges (Williams 2006, §7.4.1.1); the mapping is implemented in `src/ctrnn/genotype.py` with layout weights (row-major, positions 0–24), biases (positions 25–29), taus (positions 30–34) fixed as a tested invariant from Pass 1c.

| Parameter | Symbol | Phenotypic range |
|---|---|---|
| Weights | $w_{ji}$ | $[-10, 10]$ |
| Biases | $b_i$ | $[-10, 10]$ |
| Time constants | $\tau_{y,i}$ | $[1, 4]$ |

### 2.5 Implementation: vendored from `madvn/CTRNN`

We use the CTRNN implementation of Candadai (2020), available at `https://github.com/madvn/CTRNN` and on the module's approved software list. The implementation is small and pure-Python; we vendor its source into our repository under `src/ctrnn/_madvn.py` to remove dependency drift and to make it directly readable as part of our submitted code. The vendored copy is licensed under MIT (see `LICENSE_THIRD_PARTY`); no modifications to the integration logic have been made.

What the implementation does (relevant to our methods, described as if our own per the marking criteria):

- It stores the network state as numpy arrays for $y$, $z$, $b$, $\tau_y$, and the weight matrix $W$.
- Each call to its `euler_step(I)` method applies eq. (3) above to all nodes simultaneously, then computes eq. (2) to update $z$.
- It exposes $W$, $b$, $\tau_y$, $y$, $z$ as mutable attributes, which our HP module reads and writes between Euler steps.

The upstream source was vendored from commit `bd1b62150ab1af6d24ade69ece999e39f1f188e7` of `madvn/CTRNN` on 2026-05-15. The README 2-neuron sinusoidal oscillator example runs unchanged under numpy 2.2.4 — no patches were required. The vendored file is `src/ctrnn/_madvn.py`; the upstream MIT license is recorded in `LICENSE_THIRD_PARTY`.

The `CTRNNAgent` wrapper (`src/ctrnn/agent.py`) has been implemented with the sensor-neurons-first indexing convention (nodes 0–2 are sensor neurons, nodes 3–4 are motor neurons) enforced as a tested invariant.

We wrap the vendored class in our own `CTRNNAgent` class (`src/ctrnn/agent.py`), which adds:

- A genotype-to-phenotype mapping (see §2.4).
- A clear sensor-neuron and motor-neuron indexing convention (nodes 0–2 are sensor neurons receiving non-zero entries of $I$; nodes 3–4 are motor neurons whose firing rates drive the agent body, eq. in §5).
- A method to reset state at the start of each trial.

---

## 3. Homeostatic plasticity

### 3.1 Plastic facilitation

For each node $i$, the plastic facilitation $\rho_i \in [-1, 1]$ measures how far the node's firing rate falls outside a target homeostatic range $[H_L, H_U]$ (Di Paolo 2000; Williams 2006, §7.4.1.1):

$$
\rho_i(z_i) \;=\;
\begin{cases}
(H_L - z_i)/H_L & \text{if } z_i < H_L \\
0 & \text{if } H_L \le z_i \le H_U \\
(H_U - z_i)/(1 - H_U) & \text{if } z_i > H_U
\end{cases}
\qquad (4)
$$

When $z_i$ is below the lower bound $H_L$, $\rho_i > 0$, and synaptic scaling and intrinsic plasticity act to increase the node's input drive. When $z_i$ is above the upper bound $H_U$, $\rho_i < 0$, and they act to decrease it.

### 3.2 Synaptic scaling

Each afferent weight to node $i$ is updated as:

$$
\tau_w \dot{w}_{ji} \;=\; \rho_i \, |w_{ji}|
\qquad (5)
$$

The use of $|w_{ji}|$ rather than $w_{ji}$ ensures that both excitatory and inhibitory weights grow in magnitude when $\rho_i > 0$ and shrink when $\rho_i < 0$ — the *strength* of the input is what is regulated, not its sign.

### 3.3 Intrinsic plasticity (adaptive bias)

Each node's bias is updated as:

$$
\tau_b \dot{b}_i \;=\; \rho_i
\qquad (6)
$$

This raises the bias when the node is firing too little (shifting the sigmoid leftward and increasing $z_i$) and lowers it when the node is firing too much.

### 3.4 Parameter values

Williams Chapter 7 values, used throughout:

| Parameter | Symbol | Value |
|---|---|---|
| Lower homeostatic bound | $H_L$ | 0.2 |
| Upper homeostatic bound | $H_U$ | 0.8 |
| Synaptic scaling timescale | $\tau_w$ | 40 |
| Intrinsic plasticity timescale | $\tau_b$ | 20 |

**Note on the choice of $H_L$, $H_U$.** Williams uses $[0.25, 0.75]$ in his substrate-level analyses (Chapter 6 and Williams & Noble 2007) but $[0.2, 0.8]$ in the Chapter 7 evolvability experiments. We use the Chapter 7 values throughout, including in the substrate-level sanity check (§10.1), to keep parameter choices consistent across the project. This is a known divergence from the Chapter 6 / Williams & Noble work and is flagged in the discussion.

### 3.5 Integration

Equations (5) and (6) are integrated using Euler's forward method with the same step size $\Delta t = 0.2$ as the CTRNN itself.

---

## 4. Sensor model

`[TBD: Pass — ray sensors]`

Outline:
- Three rays in upward-facing fan, total angular span $\pi/6$ rad, mounted on the agent's periphery.
- Signal $S = S_{max}(D_{max} - D)/D_{max}$ where $D$ is distance to intersected surface, with $S_{max} = 5$ and $D_{max} = 100$ (Williams 2006, eq. 7.1).
- Ray-circle intersection geometry for ball-catching.
- Sensor signals feed CTRNN nodes 0–2 as the non-zero entries of $I$.

---

## 5. Agent body and motor model

`[TBD: Pass — agent body]`

Outline:
- Circular body of radius 5, constrained to horizontal motion.
- Position update: $\tau_x \dot{x} = z_{\text{right}} - z_{\text{left}}$, $\tau_x = 0.2$.
- $z_{\text{right}}, z_{\text{left}}$ are the firing rates of CTRNN nodes 3 and 4 respectively.

---

## 6. Environment / falling shapes

`[TBD: Pass — environment]`

Outline:
- Circles of radius 10 dropped from $y = 100$ above the agent.
- Horizontal offset uniformly drawn from 10 evenly-spaced positions in $[x_{\text{agent}} - 25, x_{\text{agent}} + 25]$.
- Horizontal velocity uniform on $[-0.3, 0.3]$; vertical velocity uniform on $[-0.5, -0.2]$.
- Shape disappears when its lowest point passes the top of the agent.
- 20 shapes per trial.

---

## 7. Fitness function

`[TBD: Pass — fitness]`

Outline:
- Williams 2006 eq. 7.3: $F = \sum_{k=1}^{N_{\text{trials}}} (1 - d_k / d_{\text{max}})$.
- $d_k$ is the horizontal distance between agent and shape centres at the moment the shape's lowest point reaches the top of the agent.
- $d_{\text{max}}$ normalises this to $[0, 1]$ (Williams uses the maximum possible $d_k$ given the spawn range).

---

## 8. Genetic algorithm

`[TBD: GA design discussion + implementation pass]`

Outline (to be settled before the GA pass):
- Asexual; mutation only.
- Population size and generations: Williams uses 50 × 500; we will likely scale to 30 × 300 (decision to be confirmed).
- Selection: microbial GA or simple truncation-with-elitism (to be decided).
- Encoding: real-valued vector of length 35, all alleles in $[-1, 1]$.
- Mutation: Gaussian perturbation with reflection at boundaries (to be confirmed against Williams' specification).
- Per-run seeding via the seeds recorded in `RunConfig`.

---

## 9. Experimental conditions

Following Williams Chapter 7 Experiments 1 and 2:

1. **No HP** — random non-plastic CTRNN; HP off throughout evolution and trials.
2. **HP during development only** — 6000 timesteps of HP before each fitness trial, then HP frozen for the trial.
3. **HP during behaviour** — HP active throughout every trial; no separate developmental phase.
4. **HP during development and behaviour** — 6000 timesteps of HP before each trial, then HP continues during the trial.

Williams uses 10 runs per condition; we will use 5 (decision documented in `design_decisions.md`).

Each evolutionary run is initialised from a different random seed, recorded in the run's manifest entry. Raw per-generation data (best genotype, fitness statistics) are saved to disk per run; all plots are regenerated from saved data, not from live runs.

---

## 10. Calibrations and sanity checks

### 10.1 Substrate-level sanity check

`[TBD: Pass — HP, after HP module is built]`

Replicates Williams (2006) Chapter 6 / Williams & Noble (2007) at small scale to confirm our HP implementation is doing what it should:

- Generate 100 random 5-node CTRNNs with parameters sampled uniformly in the phenotypic ranges (§2.4).
- For each, record the firing rates of all 5 nodes over 6000 timesteps with $I = 0$, HP off.
- Apply HP for 6000 timesteps with $I = 0$ and re-record firing rates.
- Compare distributions: HP should concentrate firing rates away from saturated tails (0 and 1) and into $[H_L, H_U]$.

This is the empirical check that justifies treating the HP module as correct for the evolvability experiments.

### 10.2 GA baseline check

`[TBD: Pass — GA]`

Outline:
- Evolve non-plastic CTRNNs on ball-catching for a small number of runs (e.g. 30 × 100).
- Confirm that fitness curves rise above the random baseline.
- Confirm that the best-evolved individuals catch most balls in a visual trajectory inspection.

---

## References

`[Compiled at end of project. Key entries already identified:]`

- Beer, R. D. (1995). On the dynamics of small continuous-time recurrent neural networks. *Adaptive Behavior*, 3(4), 469–509.
- Beer, R. D. (1996). Toward the evolution of dynamical neural networks for minimally cognitive behavior. *SAB '96*.
- Candadai, M. (2020). CTRNN [software]. https://github.com/madvn/CTRNN
- Di Paolo, E. (2000). Homeostatic adaptation to inversion of the visual field and other sensorimotor disruptions.
- Williams, H. P. (2006). *Homeostatic Plasticity in Recurrent Neural Networks: A Computational Study* (PhD thesis, University of Sussex). Chapters 6 and 7.
- Williams, H., & Noble, J. (2007). Homeostatic plasticity improves signal propagation in continuous-time recurrent neural networks. *BioSystems*, 87(2–3), 252–259.
- Stolting, J., Beer, R. D., & Izquierdo, E. J. (2023). Characterizing the role of homeostatic plasticity in central pattern generators.
- Ashby, W. R. (1960). *Design for a Brain.*
- Turrigiano, G. G. (1999). Homeostatic plasticity in neuronal networks: the more things change, the more they stay the same. *Trends in Neurosciences*.
