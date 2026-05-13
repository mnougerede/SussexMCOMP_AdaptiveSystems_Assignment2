# Experiments specification

A working document that pins down exactly what experiments will be run, what dependent and independent variables they involve, and how results will be reported. Update as the plan firms up.

---

## Experiment 1 — Substrate verification

**Purpose:** Verify that our HP implementation reproduces Williams' substrate-level result (HP improves signal propagation and oscillation likelihood). This is a sanity check, not a main result.

**Setup:** Random CTRNNs of size 5, fully connected. Generate 500 random networks. For each: test signal propagation and oscillation likelihood before HP, then after applying HP for 6000 timesteps with $I = 0$.

**Dependent variables:**
- Mean $\Delta z$ per node in response to changes in input (Williams' signal propagation metric)
- Proportion of (network, initial condition) pairs that lead to oscillations (Williams' oscillation metric)

**Independent variable:** HP applied or not.

**Expected result:** HP increases both. If not, our implementation has a bug.

**Status:** Optional sanity check. Skip if compute-budget pressure is high.

---

## Experiment 2 — Williams replication on phototaxis

**Purpose:** Reproduce Williams' core qualitative finding (HP-during-development helps; HP-during-behaviour hurts) on the moving-light phototaxis task in Sandbox.

**Setup:**

- Agent: differential-drive in Sandbox with 2 light sensors
- Network: 5-neuron fully connected CTRNN (2 sensor, 1 interneuron, 2 motor)
- Task: moving light source on a fixed trajectory (e.g. constant-velocity horizontal sweep, or sinusoidal)
- Fitness: mean negative distance to light over trial duration, normalised to $[0, 1]$
- Trial length: TBD; long enough to capture the moving light's full trajectory
- GA: population 30, 300 generations, elitism + point mutation, asexual

**Three conditions:**
1. **Baseline:** No HP. Random CTRNN, evolution proceeds directly.
2. **Developmental HP:** HP runs for 6000 timesteps with the agent stationary in the environment and motors disabled (naturalistic stationary input). Weights and biases are then frozen. Evolution proceeds on the post-development network.
3. **Online HP:** HP active throughout evolution and during all fitness trials.

**Runs:** 5 evolutionary runs per condition (different random seeds).

**Dependent variables:**
- Mean best fitness per generation (averaged across runs)
- Distribution of final fitnesses

**Plot:** fitness-vs-generation curve, three lines, error bands across runs. Box plots of final fitness per condition.

**Expected result:** Condition 2 outperforms Conditions 1 and 3. If not, this is itself interesting and worth analysing — but the qualitative replication is the gate.

---

## Experiment 3 — Developmental duration sweep (primary contribution)

**Purpose:** Quantify the relationship between developmental phase duration and post-evolution fitness. Williams' choice of 6000 timesteps is arbitrary; this experiment maps out the space.

**Setup:** As Experiment 2 Condition 2 (developmental HP, then frozen, then evolution), but with developmental phase duration as the independent variable.

**Duration values:** $\{0, 500, 1500, 3000, 6000, 12000\}$ timesteps. Note: duration 0 is the no-HP baseline; duration 6000 matches Williams.

**Runs:** 5 evolutionary runs per duration value.

**Dependent variables:**
- Mean best fitness per generation
- Final fitness (mean and distribution across runs)
- Neuron-level metrics at end of development: proportion of neurons in viable range; mean $|y|$ per neuron

**Plots:**
- Fitness-vs-generation curves, one per duration, on shared axes
- Final fitness vs duration with error bars (the headline plot)
- Neuron-in-viable-range fraction vs duration (auxiliary plot)

**Hypotheses to evaluate:**
- H1 (monotonic): more development = more benefit, saturating at some point
- H2 (threshold): there is a minimum duration below which HP provides no benefit; above it, benefit is roughly constant
- H3 (peak): too little or too much development is bad; there is an optimum
- H4 (no effect): duration does not matter once $> 0$

**Expected result:** Probably some mix of H1 and H2 — benefit rises with duration then plateaus. But H3 would be more interesting if it occurs.

---

## Experiment 4 — Secondary extension (optional)

**Pick at most one of the following, only if time permits.**

### 4a. HP timescale variation with online HP

**Setup:** Online HP condition (as Experiment 2 Condition 3) with $\tau_w, \tau_b$ varied. Williams uses $\tau_w = 40, \tau_b = 20$ — slow. Try faster ($\tau_w = 4, \tau_b = 2$) and slower ($\tau_w = 400, \tau_b = 200$).

**Question:** Does HP-during-behaviour work better when its timescale is much faster (so transients are short relative to trial length) or much slower (so HP barely changes the parameters during a trial)?

**Why interesting:** Tests the "transients" hypothesis for Williams' negative result on online HP. If faster timescales rescue online HP, the failure was a transient problem; if not, the failure is more fundamental.

### 4b. Alternative facilitation function

**Setup:** Replace the piecewise-linear $\rho$ with a smooth alternative — a Gaussian centred at $(H_L + H_U)/2$ with appropriate width. Re-run Experiments 2 and 3.

**Question:** Does the discontinuity in $\rho$'s derivative at $H_L, H_U$ contribute to its difficulty as an online rule?

### 4c. Development input regime

**Setup:** Compare three sub-conditions of Experiment 2 Condition 2: zero input ($I = 0$) vs naturalistic stationary vs random input.

**Question:** How much does the input regime during development affect the post-development network and its evolvability?

---

## Visualisation outputs per experiment

For each experiment, the following plots should be produced automatically from saved data:

1. Fitness curve(s) with error bars
2. Final fitness box plots across conditions
3. Per-neuron $z$ time series during development (representative individuals only)
4. Histogram of $z$ across neurons before vs after HP
5. Time-in-viable-range heatmap
6. Per-neuron $w$ and $b$ trajectories during development (representative individuals only)
7. For Experiment 3: final-fitness-vs-duration headline plot

---

## Compute budget estimate

Per fitness evaluation: roughly 1 second (CTRNN with a few thousand timesteps + Sandbox simulation overhead). Rough estimates:

- Experiment 1: 500 networks × 2 conditions × 1000 input tests = 1M evaluations × ~0.1s = ~30 hours. Probably skip or scale down.
- Experiment 2: 3 conditions × 5 runs × 300 gens × 30 pop × 1s = 135k seconds ≈ 37 hours.
- Experiment 3: 6 durations × 5 runs × 300 gens × 30 pop × 1s = 270k seconds ≈ 75 hours.

These are total CPU-seconds; with parallelisation across cores, much faster. But the absolute numbers indicate Experiments 2 and 3 are the budget-dominant items, and they need to be planned with care (efficient implementation, checkpointing, parallel runs).

**Mitigations:**
- Run experiments in parallel processes (one per evolutionary run)
- Reduce population or generation count if early results suggest fitness saturates earlier
- Skip Experiment 1 if substrate-level verification is not behaviourally informative
- Save raw data per run so partial completion is still usable

---

## Failure-mode protocol

If an experiment does not produce the expected result, do not panic. Possible explanations to investigate, in order:

1. **Implementation bug** — verify HP rule with unit tests, check CTRNN against reference
2. **Wrong default parameter** — check $H_L, H_U, \tau_w, \tau_b$ values
3. **Task too easy/hard** — examine baseline fitness curve; if it saturates immediately, task is too easy; if it never improves, task is too hard
4. **Trial length wrong** — if trials end before behaviour stabilises, fitness will be noisy
5. **Genuine finding** — if all of the above check out, the result may be real. Investigate and report.

The replication gate (Experiment 2) is critical. If we cannot reproduce Williams qualitatively, every downstream experiment is undermined. Spend time here.
