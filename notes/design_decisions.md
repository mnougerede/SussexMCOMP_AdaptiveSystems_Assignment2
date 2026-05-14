# Design decisions

A working document for the methodological choices in the project. Each section states the question, gives the current choice with rationale, and notes anything still open.

This document reflects the Option A decision: own Python simulator, Williams (2006) Chapter 7 ball-catching replication, three analyses as the extension.

---

## Simulator: own implementation, not Sandbox

**Decision:** Build the simulator from scratch in pure Python.

**Reasoning:** Williams' Chapter 7 evolvability experiments use ray sensors returning signal inversely proportional to distance to an intersected surface. Sandbox provides light sensors (omnidirectional or directed cones) and bump sensors but no ray or distance sensors — Chris confirms this in the Sandbox demos notes ("there is no arena or wall sensor in Sandbox, as there are no wall objects per se"). The shape discrimination task in particular requires depth profiling of the falling object across the ray fan; light sensors with shadow-casting bodies cannot produce equivalent input.

Williams' Chapter 6 photosensitive robot **is** Sandbox-compatible (two light sensors, differential drive) but Williams himself uses it only for qualitative behavioural illustration. He notes that phototaxis does not require internal dynamics, which is precisely why he reserved the quantitative evolvability experiments for the harder ray-sensor tasks. Building on Sandbox's phototaxis would make the project conceptually weaker even if implementationally easier.

The Beer-style agent is small. Beer's whole specification fits on two pages of his 1996 paper, and Williams Chapter 7 sections 7.4.1 and 7.4.2 are similarly compact. The simulator is roughly 500 lines of Python.

**Methodological benefit:** the methods section gets stronger because we own and can describe every component. Chris's Assignment 1 group feedback explicitly emphasised the methods section: "if you can't/won't explain it, you have no business using it."

---

## Task: ball-catching only, not discrimination

**Decision:** Implement and run only the ball-catching task.

**Reasoning:** Williams himself reports in Chapter 7 that the discrimination task was very difficult for HP-plastic networks; the conference paper and thesis both note discrimination performance remained poor across most conditions. Including it adds substantial implementation complexity (diamond geometry, two-shape trial generation, the discrimination fitness function) without a clear payoff: even if our results match Williams' patterns there, the patterns are weaker and harder to interpret.

**Mention in discussion:** the choice to use ball-catching only, with a note that discrimination is more demanding and was set aside for time-budget reasons. Frame discrimination as natural future work.

---

## CTRNN architecture: Williams Chapter 7 specification exactly

**Decision:** 5-node fully connected CTRNN, 3 sensor neurons and 2 motor neurons, **no interneurons**. Each ray sensor feeds a unique node. Motor output is read from the two non-sensor nodes.

**Reasoning:** This is exactly Williams' Chapter 7 specification (page 120 of the thesis). Deviating from it complicates direct comparison with Williams' results.

The absence of interneurons is conceptually striking but it's what Williams used. Mention in methods that this is a minimal architecture and that one could investigate larger networks (which Williams himself does in Chapter 7 Experiment 4).

---

## HP target range: Williams Chapter 7 values

**Decision:** $H_L = 0.2$, $H_U = 0.8$ for the evolvability experiments.

**Reasoning:** Williams uses $[0.25, 0.75]$ for the substrate-level analyses in Chapter 6 and the conference paper, but **switches to $[0.2, 0.8]$ in Chapter 7** (page 121 of the thesis). The values used should match the chapter being replicated, not the values quoted in lecture slides or the conference-paper extract. This is a small but important detail — flag it in the methods section.

---

## Plasticity timescales: Williams values

**Decision:** $\tau_w = 40$, $\tau_b = 20$.

**Reasoning:** Williams reports that both rules give similar results when applied independently. Combined rules are what he uses in the published comparisons, and these timescale values are reported as the canonical choice. Sensitivity to these is mentioned in Williams' Chapter 6 results section but not systematically explored.

---

## Sensory input during developmental phase

**Decision:** Run the developmental phase with **the agent stationary in the environment receiving real sensor input** from a fixed scene of falling shapes — same shape generation procedure as during fitness trials, but with motors disabled so the agent does not move.

**Reasoning:** Williams' thesis doesn't fully specify the developmental-phase input regime, only that "homeostatic plasticity is applied for a period in each trial prior to fitness assessment" (Chapter 7, page 119). This implies the agent is in the environment but not yet acting — the natural reading is that it receives input but doesn't move. This is also the closest to the biological analogy: HP adapts to the input distribution the network will face.

Note for methods: this choice should be stated clearly. The alternative (zero input, or random input) is a reasonable choice we considered but did not adopt.

---

## Number of evolutionary runs per condition

**Decision:** 5 runs per condition (Williams used 10), 30 population × 300 generations (Williams used 50 × 500).

**Reasoning:** Compute budget. Williams' setup is 4 conditions × 10 runs × 500 generations × 50 individuals × ~20 trials per fitness evaluation. We scale down to keep the project runnable in two weeks: 4 × 5 × 300 × 30 × (some trial count). Adequate for qualitative replication; we lose some statistical resolution but the qualitative ordering Williams observed should still be visible.

**Acknowledge in discussion:** the run count limits the statistical power of any quantitative claims. Headline results need to be robust to this limit.

---

## Performance metric: Williams' ball-catching fitness

**Decision:** Use Williams' fitness function exactly: $\sum_i (1 - d_i / d_{max})$ over trials, where $d_i$ is the horizontal distance between agent and shape centres when their leading edges meet, normalised to $[0, 1]$. Williams' thesis equation 7.3 gives the precise form.

**Reasoning:** Direct comparability with Williams' published curves. Any change to the fitness function obscures the replication.

---

## What we are NOT doing

Spelled out so we don't drift back into them:

- **No discrimination task** in the main experiments
- **No duration sweep** (Williams' Experiment 2 already addresses this with a single 6000-timestep value, and the literature has moved on)
- **No phototaxis or moving-light task** (Sandbox-driven idea that didn't survive the move to own-simulator)
- **No $\rho$ functional-form variations** in the primary experiments
- **No Hebbian plasticity comparison** (interesting future work; out of scope)

---

## Three analyses Williams did not perform (the contribution)

### Behavioural trajectory analysis

For each condition, select representative individuals (best-of-best, median-of-best, perhaps worst-of-best). Plot agent x-position and shape (x, y)-position over time across a handful of representative trials. Overlay firing rates of all 5 neurons alongside. Compare strategies qualitatively across conditions.

**Why this matters:** Chris's group feedback explicitly called out "showing only learning curves and aggregate statistics without examples of actual learned/evolved behaviours is a route to a low mark." Williams' published figures are all fitness curves; no individual behaviours are shown in the conference paper, and only minimal examples in the thesis.

### Per-neuron viable-range diagnostics across evolution

For each condition, at each generation, take the best individual; run a representative trial; record for each neuron the fraction of timesteps its firing rate is in $[H_L, H_U]$; plot the resulting per-neuron-by-generation fraction over the course of evolution.

**Why this matters:** Williams' substrate-level claims (HP keeps neurons in the viable range) and his evolvability claims (HP-during-development helps) have not been directly connected to each other in the literature. This analysis bridges them: it asks whether the substrate-level effect persists through evolution, in each condition.

### Frozen-HP test (the Stolting et al. test)

For each evolved HP-during-behaviour individual, freeze the parameters at the end of evolution and re-evaluate fitness. Compare with the HP-development-only group (where HP is already frozen for trials) as a control for measurement noise.

**Why this matters:** Stolting et al. (2023) found that some CTRNN oscillations in CPG tasks are HP-enabled — they collapse when HP is frozen. They speculated at the end of their paper that this might explain Williams' poor HP-during-behaviour results, but they did not test the claim on Williams' agent setup. We do. If HP-during-behaviour individuals show large fitness drops on freezing, the speculation is supported; if not, the poor HP-during-behaviour result must have another explanation.

This is the project's clearest single scientific question.

---

## Compute budget and parallelisation

Each fitness evaluation: 5 neurons × ~1000-2000 timesteps × ~20 shapes per trial. Approximately 0.1-0.5 seconds per evaluation depending on implementation efficiency.

Full experiment: 4 conditions × 5 runs × 300 generations × 30 individuals = 180,000 evaluations × 0.3s = ~15 hours single-threaded. Manageable on a laptop overnight, or much faster if parallelised across cores using `multiprocessing` (Williams' fitness function is trivially parallelisable across population members).

**Plan:** implement single-threaded first; parallelise if it becomes a bottleneck. Save raw data after each run so partial results are usable.

---

## Reproducibility

Each evolutionary run is initialised from a different seed. Seeds are recorded in the run output. Raw per-generation data (best genotype, fitness statistics) saved as compressed numpy arrays. All plots regenerated from saved data, not from live runs.

This was a hard lesson from Assignment 1: keep the analysis and plotting code separate from the experiment runner so that a failed plot doesn't require a re-run of the experiment.
