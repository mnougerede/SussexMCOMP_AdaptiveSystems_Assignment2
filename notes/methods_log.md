# Methods log

A working document that accrues methods-section content as the project is built. Each section starts as bulleted decisions + citations and fills in as the corresponding code is written. By the end of Phase 5, this should read as first-draft methods prose.

Conventions:
- Citations are inline as `(Williams 2006, §7.4.1.1)` style; full references compiled at the end.
- Williams quotes use the **Chapter 7** values throughout. Where Chapter 6 / conference paper values differ, this is flagged explicitly.
- Every variable in every equation is defined on the same logical block, even if redundantly.
- `[TBD: Pass N]` markers indicate content that will be written when that pass is built.
- Sections marked **(R)** are report-narrative sections (introduction or discussion); sections without are methods proper. The introduction and discussion live in the report file, not here, but their key claims and citations are tracked here so the methods log is a single source of truth for what the report says about each topic.

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

The upstream source was vendored from commit `bd1b62150ab1af6d24ade69ece999e39f1f188e7` of `madvn/CTRNN` on 2026-05-15. The README 2-neuron sinusoidal oscillator example runs unchanged under numpy 2.2.4 — no patches were required. The vendored file is `src/ctrnn/_madvn.py`; the upstream MIT license is recorded in `LICENSE_THIRD_PARTY`.

The wrapper class `CTRNNAgent` (`src/ctrnn/agent.py`) constructs from `CTRNNConfig` and establishes the sensor-neurons-first indexing convention (nodes 0–2 are sensor neurons receiving non-zero entries of $I$; nodes 3–4 are motor neurons whose firing rates drive the agent body), enforced as a tested invariant. It exposes $W$, $b$, $\tau_y$, $y$, $z$ as mutable attributes — with $W$ stored as a dense `np.ndarray` — for the HP and GA modules to read and write directly, and provides a `reset()` method that zeros all states at the start of each trial.

Note for the report: the upstream class stores weights as a scipy sparse matrix; we convert to a dense numpy array in the wrapper because (a) the network is fully connected so sparsity provides no benefit, and (b) HP performs frequent element-wise updates that are substantially faster on dense storage.

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

### 3.6 Developmental phase

In conditions where HP runs before a fitness trial (conditions 2 and 4 in §9), the network is updated for 6000 timesteps with $I = 0$. HP runs; the network state $y$ evolves. At the end of the 6000 timesteps the modified weights and biases are used for the fitness trial that follows.

**Note on the choice of input.** Williams's thesis does not specify the input regime during the developmental phase. We follow the precedent of his Chapter 6 substrate-level experiments and use zero input. This is a documented choice; the alternative reading (sensory input from a sample of falling shapes) is reasonable but Chapter 6 sets the stronger precedent.

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

`[TBD: Pass — fitness; revisit Williams eq. 7.3 carefully]`

Outline (combined displacement-reduction + final-distance score):
- Williams 2006 eq. 7.3 averaged across $N_{\text{trials}} = 10$ trials per evaluation, 20 shapes per trial.
- For each shape $k$: a displacement-reduction term measuring how much horizontal distance to the shape was reduced over the falling period, plus a final-distance term measuring proximity at the moment the shape's lowest point reaches the agent.
- Both terms normalised; total fitness is the average across trials.
- **Need to re-read Williams 7.3 carefully before implementing.** Earlier drafts described only the final-distance component; the full form has both.

---

## 8. Genetic algorithm

The GA operates over the real-valued genotype defined in §2.4: a vector of length 35 in $[-1, 1]$.

### 8.1 Selection

**Tournament selection with $K = 3$.** For each offspring slot, three individuals are drawn uniformly at random (with replacement, across the parental population) and the one with the highest fitness is selected to reproduce. This is repeated independently for each offspring.

### 8.2 Elitism

The single best individual from the parental generation is copied unchanged into the next generation. The remaining $N - 1$ offspring are produced by tournament selection followed by mutation.

### 8.3 Mutation

Each allele is mutated independently with probability $p_m$. When mutated, the new value is the old value plus a Gaussian perturbation:

$$
g_i' \;=\; g_i + \mathcal{N}(0, \sigma_m^2)
\qquad (7)
$$

Reflected at the boundaries: if $g_i' > 1$ then $g_i' \leftarrow 2 - g_i'$, and symmetrically if $g_i' < -1$. Mutation parameters $p_m$ and $\sigma_m$ to be set during the GA baseline check (Phase 6) — start with $p_m = 0.1$ and $\sigma_m = 0.1$, adjust based on whether the no-HP baseline achieves Williams's reported performance.

### 8.4 Initialisation

Each allele independently drawn uniformly from $[-1, 1]$.

### 8.5 Population, generations, runs

Target: population 50, generations 500, 10 runs per condition (Williams's scale).
Fallback if compute is constrained: 30 × 300 × 5 (decision deferred to after the GA baseline check).

### 8.6 Reproducibility

Each run uses a different RNG seed, recorded in the run's manifest entry. The seed initialises the population, the trial-sequence sampler, and the tournament-selection sampler. Resumption restores the RNG state exactly (per the persistence infrastructure tests).

### 8.7 Justification: tournament rather than Williams's roulette + elitism

Williams uses fitness-proportional roulette wheel selection with elitism of the top 5 individuals. Tournament selection is the EC textbook default for several reasons we adopt:

- **Invariance to fitness scaling.** Tournament selection depends only on the ordering of fitnesses within each tournament, not their absolute magnitudes. Fitness-proportional roulette is sensitive to fitness scaling and concentrates reproduction on the early lead when fitness variance is high.
- **Selection pressure tuning.** $K$ directly controls selection pressure; $K = 3$ gives moderate pressure suitable for continuous optimisation problems.
- **Robustness to premature convergence.** Roulette + elitism (Williams's combination) is known to cause rapid loss of population diversity, particularly when one individual gains an early fitness lead. Tournament selection avoids this failure mode.

We retain the *idea* of elitism — a single best individual is preserved per generation — to prevent fitness regressions, while avoiding the diversity collapse that elitism + roulette together produce.

**Implication for replication.** Because our GA differs from Williams's, our fitness curves will not match his quantitatively. The headline replication is at the level of *qualitative ordering of conditions*, not at the level of matching specific curves. This is appropriate: the project's central claims are about HP dynamics, not GA dynamics. If our qualitative ordering differs from Williams's, the difference is itself a substantive finding and is treated as such.

---

## 9. Experimental conditions

Following Williams Chapter 7 Experiments 1 and 2:

1. **No HP** — random non-plastic CTRNN; HP off throughout evolution and trials.
2. **HP during development only** — 6000 timesteps of HP with $I = 0$ before each fitness trial, then HP frozen for the trial.
3. **HP during behaviour** — HP active throughout every trial; no separate developmental phase.
4. **HP during development and behaviour** — 6000 timesteps of HP before each trial, then HP continues during the trial.

Williams uses 10 runs per condition; we target 10, fall back to 5 if compute requires (decision documented in `notes/design_decisions.md`).

Each evolutionary run is initialised from a different random seed, recorded in the run's manifest entry. Raw per-generation data (best genotype, fitness statistics) are saved to disk per run; all plots are regenerated from saved data, not from live runs.

---

## 10. Calibrations and sanity checks

### 10.1 Substrate-level sanity check

The check generated 100 random 5-node CTRNNs (weights and biases sampled uniformly from $[-10, 10]$, time constants from $[1, 4]$, seed 42). For each network, firing rates were recorded over 220 timesteps ($I = 0$, HP off) to establish a baseline, then HP was applied for 6000 timesteps ($I = 0$, $H_L = 0.2$, $H_U = 0.8$, $\tau_w = 40$, $\tau_b = 20$, $\Delta t = 0.2$). Firing rates were recorded during the final 220 HP steps and again for 220 steps after HP was switched off (continuing from the post-HP neural state).

Results: fraction of firing-rate samples outside $[H_L, H_U]$ was 0.861 before HP, 0.477 during HP (final 220 steps), and 0.539 after HP was switched off. The improvement from before to during HP confirms that the plasticity rules are functioning as intended. The partial persistence of improvement after HP is switched off shows that HP has genuinely reshaped weights and biases, not merely masked saturation dynamically. The residual gap between during and after suggests that a portion of HP's regulatory work depends on ongoing parameter adjustment — a substrate-level instance of the HP-enabled dynamic that the frozen-HP test (§10.3) will probe in evolved networks. Figure: `figs/substrate_check.pdf`.

### 10.2 GA baseline check

`[TBD: Pass — GA]`

Outline:
- Evolve non-plastic CTRNNs on ball-catching for a small number of runs (e.g. 30 × 100).
- Confirm that fitness curves rise above the random baseline.
- Confirm that the best-evolved individuals catch most balls in a visual trajectory inspection.
- Measure single-evaluation runtime; use this to decide between Williams-scale and scaled-down for the main experiments.

---

## 11. (R) Framing for introduction and discussion

This section tracks key claims and citations for the introduction and discussion sections of the report. The prose lives in the report file; this is a stable record of what claims it makes and what it cites for them.

### 11.1 The Baldwin effect frame

**Claim.** This project's central question is best framed via the Baldwin effect: lifetime adaptation (HP) can guide evolution, and lifetime-acquired traits can become genetically assimilated over generations. The four experimental conditions vary the kind and amount of lifetime adaptation; the frozen-HP test directly probes whether genetic assimilation has occurred.

**Key citations.**
- Baldwin, J. M. (1896). A new factor in evolution. *American Naturalist*, 30(354), 441–451.
- Hinton, G. E., & Nowlan, S. J. (1987). How learning can guide evolution. *Complex Systems*, 1(3), 495–502. — canonical introduction of the Baldwin effect in evolutionary computation.
- Mayley, G. (1996). Landscapes, learning costs, and genetic assimilation. *Evolutionary Computation*, 4(3), 213–234. — formal conditions under which genetic assimilation occurs in EC.
- Turney, P. (1996). Myths and legends of the Baldwin effect. *ICML 1996 Workshop on Evolutionary Computation and Machine Learning*. — useful as a reality check on what the Baldwin effect does and does not predict.

**Where this lives in the report.**
- *Introduction.* The Baldwin effect frame is introduced after the working definition of adaptivity and before the substrate-level/evolvability tension. The connection to the project: HP-during-development is the Hinton-&-Nowlan setup; HP-during-behaviour adds the complication that lifetime adaptation and fitness measurement are interleaved.
- *Discussion.* The frozen-HP test result is interpreted in terms of degree of genetic assimilation. Large fitness drop on freezing → assimilation has not occurred, HP-during-behaviour evolution remains plasticity-dependent. Small drop → assimilation has occurred, evolved genotypes encode the solution innately.

### 11.2 Search dynamics, not "moving fitness landscape"

The fitness landscape (the genotype → fitness mapping) is determined by the evaluation procedure and is deterministic for a given seed regardless of plasticity. What HP changes is the *phenotype that gets evaluated*: the genotype encodes initial network parameters, and HP modifies them before fitness is measured. This is a feature of the encoding (indirect, via HP), not of the landscape (whose shape is unchanged).

The proposal used the wrong phrasing ("moving fitness landscape"). The correct phrasing is "HP changes the search dynamics by introducing an indirect genotype-phenotype mapping", or equivalently "HP changes the effective phenotype that the GA selects on".

### 11.3 The Stolting hypothesis as mechanism

Stolting, Beer & Izquierdo (2023) speculated that HP-during-behaviour evolves limit cycles in the joint network-and-parameter state space — dynamics in which both the network state and the HP-modified parameters co-vary in a way that is essential to behaviour. Such dynamics would collapse if HP were frozen. The frozen-HP test in our project bears directly on this: a large fitness drop is consistent with Stolting's mechanism, a small drop is inconsistent with it.

The Baldwin frame and Stolting frame are complementary: Stolting proposes a specific mechanism by which genetic assimilation can fail (HP creates dynamics that cannot be reproduced without it); the Baldwin frame is the more general claim that assimilation either has or has not occurred. The same frozen-HP test answers both.

---

## References

`[Compiled at end of project. Key entries identified so far:]`

- Ashby, W. R. (1960). *Design for a Brain* (2nd ed.). Chapman & Hall.
- Baldwin, J. M. (1896). A new factor in evolution. *American Naturalist*, 30(354), 441–451.
- Beer, R. D. (1995). On the dynamics of small continuous-time recurrent neural networks. *Adaptive Behavior*, 3(4), 469–509.
- Beer, R. D. (1996). Toward the evolution of dynamical neural networks for minimally cognitive behavior. *SAB '96*.
- Candadai, M. (2020). *CTRNN: A Python package for the simulation of continuous-time recurrent neural networks* [software]. https://github.com/madvn/CTRNN
- Di Paolo, E. A. (2000). Homeostatic adaptation to inversion of the visual field and other sensorimotor disruptions.
- Hinton, G. E., & Nowlan, S. J. (1987). How learning can guide evolution. *Complex Systems*, 1(3), 495–502.
- Mayley, G. (1996). Landscapes, learning costs, and genetic assimilation. *Evolutionary Computation*, 4(3), 213–234.
- Stolting, J., Beer, R. D., & Izquierdo, E. J. (2023). Characterizing the role of homeostatic plasticity in central pattern generators.
- Turney, P. (1996). Myths and legends of the Baldwin effect. *ICML 1996 Workshop on Evolutionary Computation and Machine Learning*.
- Turrigiano, G. G. (1999). Homeostatic plasticity in neuronal networks: the more things change, the more they stay the same. *Trends in Neurosciences*, 22(5), 221–227.
- Williams, H. P. (2006). *Homeostatic Adaptive Networks*. PhD thesis, University of Leeds.
- Williams, H., & Noble, J. (2007). Homeostatic plasticity improves signal propagation in continuous-time recurrent neural networks. *BioSystems*, 87(2–3), 252–259.
