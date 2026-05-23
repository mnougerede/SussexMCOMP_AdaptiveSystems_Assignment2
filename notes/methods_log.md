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

Three upward-pointing rays are mounted on the agent, arranged in a fan of total angular span $\pi/6$ rad centred on vertical. The ray angles from vertical are $-\pi/12$, $0$, and $+\pi/12$ for the left, centre, and right sensors respectively. For each ray, the distance $D$ to the nearest intersection with a falling shape (treated as a circle) is computed using the standard ray-circle intersection formula. The perpendicular distance $d_\perp$ from the shape centre to the ray line is computed, and an intersection exists when $\Delta = r^2 - d_\perp^2 \geq 0$; the nearest positive intersection distance is $D = t_{\text{proj}} - \sqrt{\Delta}$. If no intersection exists, or $D > D_{\max} = 100$, the sensor returns zero. Otherwise the signal is (Williams 2006, eq. 7.1):

$$S_i = S_{\max} \frac{D_{\max} - D}{D_{\max}} \qquad (8)$$

with $S_{\max} = 5$. The three sensor signals $S_0, S_1, S_2$ are injected as the external input to CTRNN nodes 0, 1, 2 respectively (i.e. $I_i = S_i$ for $i \in \{0,1,2\}$; $I_i = 0$ for $i \in \{3,4\}$).

---

## 5. Agent body and motor model

The agent is a circle of radius 5, constrained to horizontal motion at a fixed vertical position. Its horizontal position $x$ evolves according to:

$$\tau_x \dot{x} = z_{\text{right}} - z_{\text{left}} \qquad (9)$$

where $z_{\text{right}}$ and $z_{\text{left}}$ are the firing rates of CTRNN nodes 4 and 3 respectively, and $\tau_x = 0.2$. Integrated with Euler's method at $\Delta t = 0.2$: $x(t + \Delta t) = x(t) + (z_{\text{right}} - z_{\text{left}})$. The agent position is unbounded horizontally; there are no walls.

---

## 6. Environment / falling shapes

Each trial presents 20 falling shapes sequentially. Each shape is a circle of radius 10, spawned at $y = 100$ above the agent's fixed vertical position with a horizontal offset drawn uniformly from $[-25, 25]$ relative to the agent's current position. Horizontal velocity $v_x$ is drawn uniformly from $[-0.3, 0.3]$ and vertical velocity $v_y$ from $[-0.5, -0.2]$. Shapes fall under constant velocity (no acceleration). A shape is considered to have passed when its lowest point ($y - r$) drops below the top of the agent ($0 + r_{\text{agent}} = 5$), at which point the next shape is spawned. The random state for shape generation is seeded per trial and recorded in the `TrialRecord` for exact reproducibility.

---

## 7. Fitness function

`[TBD: Pass — fitness function; implement after re-reading Williams eq. 7.3]`

Key constraint: Williams eq. 7.3 is a combined score with a displacement-reduction term and a final-distance term. Both components must be included. Do not implement from memory — re-read §7.4.1.2 of the thesis before writing any code.

---

## 8. Genetic algorithm

The GA operates over the real-valued genotype defined in §2.4: a vector of length 35 in $[-1, 1]$.

### 8.1 Selection

**Tournament selection with $K = 3$.** For each offspring slot, three individuals are drawn uniformly at random (with replacement, across the parental population) and the one with the highest fitness is selected to reproduce. This is repeated independently for each offspring.

### 8.2 Elitism

The single best individual from the parental generation is copied unchanged into the next generation. The remaining $N - 1$ offspring are produced by tournament selection followed by mutation.

### 8.3 Mutation

Each allele is mutated independently with probability $p_m = 0.03$ (Williams's rate). When mutated, the new value is the old value plus a Gaussian perturbation:

$$
g_i' \;=\; g_i + \mathcal{N}(0, \sigma_m^2)
\qquad (7)$$

with $\sigma_m = 0.1$. Reflected at the boundaries: if $g_i' > 1$ then $g_i' \leftarrow 2 - g_i'$, and symmetrically if $g_i' < -1$.

Williams uses a mixed mutation scheme (half uniform reset, half uniform perturbation). We use a pure Gaussian scheme with Williams's rate. The Gaussian concentrates mutations near the current value, giving stronger local search properties than the flat uniform perturbation, while the Gaussian tail provides occasional larger jumps. $p_m = 0.03$ matches Williams and gives on average one allele mutated per genotype per generation (35 alleles × 0.03 ≈ 1.05).

### 8.4 Initialisation

Each allele independently drawn uniformly from $[-1, 1]$.

### 8.5 Population, generations, runs

Target: population 20, generations 200, 5 runs per condition. Williams used 50 × 500 × 10, which exceeds the available compute budget at our per-evaluation runtime. The scaled-down parameters are documented in `plan/experiment_targets.json`. Expected wall time on the desktop (i5-9600K, 6 cores, `n_workers=6`): approximately 1 hour per condition.

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

Williams uses 10 runs per condition; we use 5 (documented in `plan/experiment_targets.json`; see `notes/design_decisions.md` for compute rationale).

Each evolutionary run is initialised from a different random seed, recorded in the run's manifest entry. Raw per-generation data (best genotype, fitness statistics) are saved to disk per run; all plots are regenerated from saved data, not from live runs.

---

## 10. Calibrations and sanity checks

### 10.1 Substrate-level sanity check

The check generated 100 random 5-node CTRNNs (weights and biases sampled uniformly from $[-10, 10]$, time constants from $[1, 4]$, seed 42). For each network, firing rates were recorded over 220 timesteps ($I = 0$, HP off) to establish a baseline, then HP was applied for 6000 timesteps ($I = 0$, $H_L = 0.2$, $H_U = 0.8$, $\tau_w = 40$, $\tau_b = 20$, $\Delta t = 0.2$). Firing rates were recorded during the final 220 HP steps and again for 220 steps after HP was switched off (continuing from the post-HP neural state).

Results: fraction of firing-rate samples outside $[H_L, H_U]$ was 0.861 before HP, 0.477 during HP (final 220 steps), and 0.539 after HP was switched off. The improvement from before to during HP confirms that the plasticity rules are functioning as intended. The partial persistence of improvement after HP is switched off shows that HP has genuinely reshaped weights and biases, not merely masked saturation dynamically. The residual gap between during and after suggests that a portion of HP's regulatory work depends on ongoing parameter adjustment — a substrate-level instance of the HP-enabled dynamic that the frozen-HP test (§10.3) will probe in evolved networks. Figure: `figs/substrate_check.pdf`.

### 10.2 GA baseline check

`[TBD: Pass — run ga_baseline_check.py on desktop after WSL2 setup]`

Script is ready (`scripts/ga_baseline_check.py`). Will evolve non-plastic CTRNNs for 100 generations, population 30, seed 42, `n_workers=6` on the desktop. Gate: fitness curve rises clearly above 0.5 by generation 50–100; trajectory shows agent tracking shapes; per-evaluation timing confirms the full experiment is feasible overnight.

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

### 11.4 Interpreting the condition results: a hypothesis and an open puzzle

This subsection records the working interpretation of the four-condition results, flagged as hypothesis rather than established mechanism. The viable-range diagnostics (8b) and the frozen-HP test (8c) are the analyses that will test it; until they are done, the prose in the report should present this as a question the data raises, not a conclusion it supports.

**What the evidence currently shows.** Two findings are solid. First, from the search-dynamics figure: the Dev only condition converges to a higher final fitness than the other three while collapsing its population fitness spread on the same early timescale (by roughly generation 10 to 20) as the others. It does not search longer or maintain more diversity; it converges to a better optimum. Second, from the replication statistics: Both is statistically indistinguishable from the No HP baseline (Kruskal-Wallis across conditions significant at p = 0.027, driven by Dev only; No HP, Behaviour only and Both mutually indistinguishable at n = 5).

**The hypothesis for Dev only.** The developmental phase acts as a within-lifetime adaptation that pre-conditions each individual before evaluation, so the genotype is selected for its fitness *after development* rather than for its innate fitness. The GA still converges quickly, but the optimum it converges onto is higher because every individual starts evaluation from a better-developed phenotype. This is the Baldwin setup: selection acts on the developed phenotype, and acquired developmental changes are not inherited (the genotype is unchanged between generations); the benefit propagates only through the selective advantage that good developmental potential confers. This framing connects directly to Williams's adiabatic-elimination point and to Hinton & Nowlan.

**The open puzzle for Both.** The simple "developmental phase gives a better starting point" story does not, on its own, explain Both. Both runs the identical developmental phase to Dev only, so it should inherit the same head start; the only difference is that Both leaves HP active during the trial whereas Dev only freezes it. Yet Both is dragged back to the No HP baseline. Whatever benefit the developmental phase confers is therefore cancelled by HP remaining active during behaviour. This is not a flaw in the Dev only hypothesis so much as evidence that it is incomplete: the developmental benefit is real (Dev only shows it) but fragile to ongoing plasticity during behaviour (Both shows it). The mechanism of that cancellation is exactly what the frozen-HP test is positioned to probe. If Both individuals depend on ongoing HP in a way that disrupts the developmentally-established behaviour, freezing HP at the end of development could recover fitness rather than reduce it, which would be a notable result. The report should not assert the "better starting point" mechanism for Dev only without also addressing why it fails for Both, and should defer the mechanistic claim until 8b and 8c are in.

**Note on premature convergence.** The early collapse of population fitness spread in all four conditions is not treated as a GA pathology. With a population of 30 on a 35-dimensional continuous genotype, sustained population diversity was not expected; the GA locates a basin quickly and then refines within it. Fitness continues to improve after the spread collapses, which indicates exploitation within a converged gene pool rather than a stalled search. For an evolvability study this is acceptable and arguably useful: a GA that converges to a range of different optima across conditions is what makes HP's effect on the search visible. The relevant point for the discussion is that the differences between conditions arise from *where* each converges, not from differences in whether or how fast they converge.

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
