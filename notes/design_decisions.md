# Design decisions

A working document for the methodological choices that need to be made for this project. Each section states the question, summarises the options, gives a current best guess, and flags whether it needs to be resolved before the proposal, before implementation, or before writing.

---

## Task choice

**Question:** What task should the Sandbox agent perform?

**Constraint:** The task needs to be non-trivial enough that internal CTRNN dynamics (oscillations, integration over time) actually matter for performance. Pure phototaxis is too easy — a Braitenberg vehicle solves it without any learning, so HP cannot demonstrably help.

**Options considered:**

| Task | Pros | Cons |
|---|---|---|
| Static phototaxis | Trivial to set up; standard Sandbox | Too easy; HP cannot show benefit |
| Phototaxis with moving light | Requires temporal prediction; one fitness number; parametrically scalable difficulty | Implementation requires custom light motion |
| Two-light discrimination | Closer to Williams' shape discrimination | Task design needs care; may end up easy |
| Phototaxis with sensor inversion (Lab 4-style) | Uses existing Sandbox infrastructure | Adaptation question entangles with HP question |
| Light-seeking with obstacles | Genuinely needs internal state for navigation | Complex Sandbox setup |

**Current choice:** Phototaxis with a moving light source. Primary because (a) it is the cleanest minimal extension of standard phototaxis that requires temporal prediction, (b) it is parametrically scalable in difficulty, (c) it is implementable in Sandbox without heavy custom infrastructure.

**Status:** To be discussed with Chris in proposal email. Frame as "I considered the alternatives X, Y, Z; I propose moving light because of A, B, C; happy to consider alternatives".

---

## Sensory input during the developmental phase

**Question:** During the 6000-timestep developmental phase, what input is the CTRNN receiving?

**The problem:** Williams (2005) does not specify. The lecture slides do not specify. The 2007 paper may or may not — has not been read yet.

**Why it matters:** HP adapts the network's parameters in response to the firing rates the network is exhibiting. The firing rates depend on the input. Different input regimes during development will produce different post-development networks.

**Options:**

| Input regime | Description | Theoretical implication |
|---|---|---|
| Zero input | $I = 0$ for all sensors throughout development | HP settles network based purely on internal dynamics; most conservative choice |
| Random input | $I$ drawn from a uniform distribution each timestep | HP averages over input distribution; closer to ensemble-level finding in Sections IV |
| Naturalistic stationary | Agent placed in environment, motors disabled, receives real sensor readings | HP adapts to input distribution it will actually face during behaviour |
| Embodied developmental motion | Agent moves randomly during development with motors active | HP adapts to self-generated sensorimotor coupling |

**Current choice:** Naturalistic stationary input — agent in environment, motors disabled, receiving real sensor input. Rationale: HP adapts to the actual input distribution the network will face during behaviour, which is the most natural reading of "before behaviour".

**Optional extension:** Compare with zero input as a control condition. If results differ qualitatively, this is itself a finding worth reporting.

**Status:** Resolve before implementation phase. Mention in methods section. Worth one sentence in the proposal.

---

## CTRNN size and topology

**Question:** How many neurons? Fully connected or sparse? Sensor and motor mappings?

**Williams' choice:** 5 neurons, fully connected, 3 sensor neurons and 2 motor neurons, no interneurons.

**Constraints for our agent:** We are using a 2-light-sensor differential-drive Sandbox agent. So 2 sensor neurons and 2 motor neurons minimum.

**Current choice:** 5 neurons, fully connected. 2 sensor neurons (one per light sensor), 2 motor neurons (one per wheel), 1 interneuron. This is the minimum that lets the network have internal capacity beyond direct sensor-motor coupling, and matches Williams' size.

**Status:** Default. Can be revisited if it does not work.

---

## Performance / fitness metric

**Question:** What does "good behaviour" mean numerically?

**Williams' choice:** Mean fitness over 10 trials; normalised to $[0, 1]$. For ball catching: proportion of objects caught. For discrimination: catch circles, avoid diamonds.

**For moving-light phototaxis, options:**

| Metric | Description |
|---|---|
| Final distance | Distance to light at end of trial |
| Mean distance | Distance to light averaged over trial duration |
| Integrated negative distance | Sum of (max_distance − current_distance) over time; rewards being close longer |
| Time within radius | Fraction of timesteps the agent is within $r$ of the light |

**Current choice:** Mean negative distance (lower is better), normalised so 1.0 = perfect tracking and 0.0 = no improvement over a stationary agent. This rewards both reaching the light and tracking it consistently.

**Status:** Resolve before Milestone 3 (replication). Probably written into proposal methods section.

---

## Evolvability vs. trained performance — what we are measuring

**Question:** Is the dependent variable evolvability (fitness over generations) or post-evolution task performance?

**Williams' choice:** Evolvability — best fitness in population across 500 generations, plotted as a curve.

**Considerations:**

- Evolvability is more aligned with Williams' framing and with the module's evolutionary-robotics material.
- Post-evolution performance is a different question: "given an already-good network, does HP refine it?" — which is not what Williams asked.
- Evolvability is more expensive to measure (need to run a full GA, multiple times, for each condition).

**Current choice:** Evolvability as primary measure. Compute budget allowing.

**Compute budget note:** With $N$ duration sweep values $\times$ 2 conditions (with/without HP) $\times$ $K$ evolutionary runs $\times$ 500 generations $\times$ 50 population $\times$ trial length, total cost scales as $N \cdot K \cdot 25000 \cdot \text{trial cost}$. For $N = 6$, $K = 5$, this is 750k fitness evaluations. Achievable but needs efficient implementation.

**Status:** Resolved. Implementation must be designed for efficient batched fitness evaluation.

---

## GA design

**Question:** What evolutionary algorithm specifics?

**Williams' choice:** Population 50, 500 generations, elitism + point mutation, no crossover, asexual. 10 runs per condition.

**Current choice:** Match Williams qualitatively but reduce numbers to fit compute budget. Population 30, 300 generations, elitism + point mutation. 5 runs per condition. To be revisited based on Sandbox/`stochsearch` infrastructure.

**Status:** Resolve when reading Lab 6/7 materials and `stochsearch` documentation.

---

## Transient handling

**Question:** When testing fitness, how do we handle the transient period during which a plastic CTRNN's dynamics are not yet settled?

**Options:**

| Approach | Pros | Cons |
|---|---|---|
| Fixed development duration (Williams' approach) | Simple; matches paper; comparable across conditions | Arbitrary; may overshoot or undershoot equilibrium |
| Equilibrium-detection-then-test | More principled; ensures fair comparison | Implementation messier; needs robust convergence test |
| Discard initial $N$ timesteps of fitness evaluation | Simple compromise | Still arbitrary |

**Current choice:** Fixed duration for primary results (matches Williams). Equilibrium detection as an optional secondary analysis ("how does actual equilibrium time vary with duration, and does it match Williams' arbitrary 6000?").

**Status:** Resolved for primary results.

---

## CTRNN implementation: from scratch or library?

**Question:** Write our own CTRNN or use `madvn/CTRNN`?

**Current choice:** Write our own. Reasons: total control, transparency, easy to add HP. Use `madvn/CTRNN` as a reference. Approximately 50 lines of Python.

**HP rule:** Definitely write our own. There is no library specifically for the Williams rule.

**Agent / environment:** Use Sandbox.

**GA:** Use `stochsearch` or Sandbox-provided GA, supplemented or replaced if needed.

**Status:** Resolved.

---

## Visualisation strategy

**Question:** What standard visualisations do we want from every experiment?

**Plan:**

1. Per-trial: time series of $z$ for each neuron, with $[H_L, H_U]$ shaded. Time series of $w$ (one line per afferent connection) and $b$ for each neuron.
2. Per-condition: histogram of $z$ across all neurons over all timesteps. Heat map of "time spent in viable range" per neuron.
3. Per-experiment: fitness curve over generations, with error bars across runs. Box plots of final fitness across conditions.
4. Optional: live overlay of $\rho(z)$ curve during development with current $z$ marked. Useful for intuition-building and demos.
5. Optional: equations, definitions, and live parameter values as sidebar panel in main visualisation. Useful for demonstration.

**Status:** Build into simulation infrastructure from the start.

---

## Reproducibility

**Question:** How to ensure experiments can be re-run and pre-empted compute does not lose work?

**Plan:** Use a deterministic seed system. Save raw data (per-trial firing rates, weights, biases, fitnesses) to disk in a structured format (e.g. compressed `.npz` per run). All plots generated from saved data, not live. Configuration files for each experiment. Logs of all runs.

**Status:** Build in from the start. Lessons from Assignment 1 apply.
