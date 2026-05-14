# Assignment 2 — CTRNN Homeostatic Plasticity: To-Do List

**Deadline:** 28 May 2026. Two other assignments running in parallel. Roughly two weeks available.

This is the live to-do list. Re-check it at the start of each working session.

---

## Project shape

**Headline:** Replicate Williams (2006) Chapter 7 ball-catching evolvability experiments with our own Python implementation of a Beer-style ray-sensor agent. Extend with three analyses Williams did not perform: behavioural trajectory inspection, per-neuron viable-range diagnostics across evolution, and a frozen-HP test on HP-during-behaviour individuals (testing the Stolting, Beer & Izquierdo 2023 HP-enabled-oscillation hypothesis in Williams' setting).

**Why not Sandbox:** Williams' Chapter 7 evolvability experiments use ray sensors returning signal inversely proportional to distance to intersected surface. Sandbox provides light and bump sensors but no ray/distance sensors. The Chapter 6 photosensitive robot is Sandbox-compatible but Williams uses it only for qualitative illustration because phototaxis does not need internal dynamics — it is the wrong task to study HP's behavioural effects quantitatively. Building the Beer-style agent in pure Python is the right methodological choice, and the methods section becomes stronger for owning the whole simulator.

**Why this extension is meaningful:** Stolting et al. (2023) found that HP can create *HP-enabled oscillations* in small CTRNNs — limit cycles that traverse the extended state space of changing weights and biases, which collapse when HP is frozen. They speculate one sentence at the end that this might explain Williams' poor HP-during-behaviour results: tasks like ball-catching may require equilibrium dynamics that HP prevents. **Nobody has tested this on Williams' agent.** Our frozen-HP test is exactly that test.

---

## Lessons from Assignment 1 feedback

Plan these in from the start, not retrofit at the end.

1. **Methods is the section that needs the most work this time.** Every CTRNN equation, every plasticity equation, every symbol defined on the page. Sensor model, environment parameters, GA, initialisation — all explicit. Symbol audit pass before submission.
2. **System diagram.** Components and couplings (CTRNN ↔ body ↔ environment, with HP as a separate adaptive process). Sketch early.
3. **Behavioural examples are an explicit deliverable**, not a nice-to-have.
4. **Definition-of-adaptivity thread.** Cite in introduction (Di Paolo grounded in Ashby), apply to the experiment, return in discussion.
5. **Wider-context threads pre-planned.** HP as cybernetic negative feedback; ontogenetic vs lifetime adaptation; HP-enabled oscillations as a dynamical-systems framing.
6. **Lit review substantial and integrated**, not a separate block. Aim for ~10 entries in the bibliography.

---

## Phase 0 — Grounding (done)

- [x] Read Williams 2005 conference paper
- [x] Read Williams 2006 thesis Chapters 6 and 7 (the evolvability material)
- [x] Read Williams & Noble 2007 (substrate-level follow-up)
- [x] Read Beer 1996 (the original ball-catching agent specification)
- [x] Read Stolting, Beer & Izquierdo 2023 (HP-enabled oscillations)
- [x] Confirm canonical equation forms; sign-check the plasticity rule

---

## Phase 1 — Project setup (mostly done)

- [x] Create GitHub repo `SussexMCOMP_AdaptiveSystems_Assignment2`
- [x] Clone locally; run `bootstrap.sh`; drop notes and planning documents into folders
- [x] First commit and push
- [ ] **Sketch the system diagram now, on paper.** Components: agent body, ray sensors, falling shape, CTRNN, HP module, GA. Couplings: sensorimotor loop, parameter-update loop, GA selection loop. Save into `notes/system_diagram_sketch.png`.

---

## Phase 2 — Submit proposal

- [x] Re-read existing proposal draft and recognise its problems
- [ ] Rewrite proposal content (Claude to provide new version reflecting Option A — see separate document)
- [ ] Transfer rewritten content into `proposal_form.docx`
- [ ] Send to Chris with a minimal cover email

---

## Phase 3 — Build the simulator core

This is two to three days of work, mostly mechanical. The whole simulator is small (~500 lines).

- [ ] CTRNN class
	- [ ] State equation: $\tau_y \dot{y} = -y + \sum w_{ji} z_j + I$
	- [ ] Activation: $z = \sigma(y + b)$
	- [ ] Euler integration with step 0.2
	- [ ] Parameter setters and getters that HP can call into
	- [ ] Unit tests: zero input + zero weights → decay to zero; constant input → stable equilibrium; sanity check against `madvn/CTRNN` on a few worked examples
- [ ] Ray sensors
	- [ ] Three sensors mounted on agent periphery, upward-facing fan spanning $\pi/6$ radians
	- [ ] Signal $S = S_{max}(D_{max} - D)/D_{max}$ where $D$ is distance to intersected surface (Williams 2006 eq. 7.1, with $S_{max} = 5, D_{max} = 100$)
	- [ ] Ray-circle and ray-line-segment intersection (line segments for diamond edges)
	- [ ] Unit tests with known geometry
- [ ] Agent body
	- [ ] Circular, radius 5
	- [ ] Horizontal motion only: $\tau_x \dot{x} = z_{right} - z_{left}$, $\tau_x = 0.2$
- [ ] Falling shapes
	- [ ] Circles (radius 10) and diamonds (vertices on radius-10 circle)
	- [ ] Drop from height 100 above agent, horizontal offset randomly chosen from 10 positions evenly distributed in $[x_{agent} - 25, x_{agent} + 25]$
	- [ ] Vertical velocity is task-parameter
	- [ ] Disappear when lowest point passes uppermost part of agent
- [ ] Environment / trial runner
	- [ ] Sequential drop of $N$ shapes; record per-timestep agent position, neural state, shape position
- [ ] Visualisation
	- [ ] Static trajectory plot (agent x, shape x and y, over time)
	- [ ] Neural state subplots alongside
	- [ ] Optional: animation for debugging and presentations

**Gate:** I can construct a hand-coded controller that catches circles and verify the simulator runs sensibly.

---

## Phase 4 — Homeostatic plasticity

- [ ] HP module as a standalone class operating on a CTRNN instance
	- [ ] Plastic facilitation $\rho(z)$ per neuron
	- [ ] Synaptic scaling: $\tau_w \dot{w} = \rho |w|$ for each afferent weight
	- [ ] Intrinsic plasticity: $\tau_b \dot{b} = \rho$
	- [ ] Parameters: $H_L = 0.2, H_U = 0.8$ (Williams' Chapter 7 values, **not** the 0.25/0.75 from his earlier chapters), $\tau_w = 40, \tau_b = 20$
- [ ] Unit tests
	- [ ] $z = 0.1 \Rightarrow \rho > 0$; bias rises; $|w|$ rises
	- [ ] $z = 0.5 \Rightarrow \rho = 0$; no change
	- [ ] $z = 0.9 \Rightarrow \rho < 0$; bias falls; $|w|$ falls
	- [ ] Excitatory and inhibitory weights both shrink under $\rho < 0$
	- [ ] Boundary values $z = H_L$ and $z = H_U$ give $\rho = 0$
- [ ] Substrate-level sanity check
	- [ ] Generate 100 random 5-node CTRNNs; histogram firing rates with HP off
	- [ ] Apply HP for 6000 timesteps with $I = 0$; histogram firing rates with HP on
	- [ ] Should see concentration shift away from saturated tails

**Gate:** HP demonstrably moves neurons out of saturation in random networks.

---

## Phase 5 — Genetic algorithm and Williams' replication setup

- [ ] GA implementation (microbial or simple elitist + mutation)
	- [ ] Genotype: real-valued vector of length $N^2 + 2N$ encoding all weights, biases, decay constants (35 values for 5 nodes)
	- [ ] Allele range $[-1, 1]$ linearly mapped to phenotypic parameter ranges
	- [ ] Williams used population 50, 500 generations; we can scale to 30 × 300 if compute is tight
	- [ ] Asexual; point mutation only
	- [ ] Reasonable seeding and reproducibility
- [ ] Fitness functions
	- [ ] Ball-catching: see Williams eq. 7.3 — sum of (1 - d) over trials, where d is normalised distance at catch
	- [ ] Skip discrimination for now — Williams himself found this task too hard for HP-plastic networks to do well; adds risk without payoff. Mention in discussion as future work.
- [ ] Single-condition test run
	- [ ] Evolve non-plastic CTRNNs on ball-catching for a small number of runs
	- [ ] Confirm fitness curves rise from generation 0
	- [ ] Confirm best-evolved individuals catch most balls
- [ ] Save raw data per run (genotypes, fitnesses per generation) to disk

**Gate:** evolution works on the baseline condition.

---

## Phase 6 — Williams replication (Milestone)

Four conditions, following Williams Chapter 7 Experiment 1 and Experiment 2:

1. **No HP** — random non-plastic CTRNN, evolution proceeds directly
2. **HP during development only** — HP for 6000 timesteps before each fitness trial, then frozen for the trial; HP off during the rest of evolution as well
3. **HP during behaviour** — HP active throughout every trial; no separate developmental phase
4. **HP during development and behaviour** — HP for 6000 timesteps before each trial, then continues during the trial

- [ ] Run 5 evolutionary runs per condition (Williams used 10; we can scale down if compute is tight)
- [ ] Best-fitness-per-generation curve per condition, with error bands across runs (Williams Fig. 7.2 equivalent)
- [ ] Mean fitness over a generation window for the final state, box plots across runs
- [ ] Qualitative check: does Williams' ordering hold? — developmental-only and no-HP roughly equivalent in *final* fitness, with developmental showing faster early progress; HP-during-behaviour worse

**Gate:** Williams' core qualitative result reproduced. This is the headline replication regardless of what comes next.

---

## Phase 7 — The three analyses Williams did not perform (Milestone)

This is the contribution.

### 7a. Behavioural trajectory analysis

- [ ] Pick representative individuals from each condition (e.g. best-of-best, median-of-best, worst-of-best)
- [ ] For each, plot agent x and shape x over time across a small number of representative trials
- [ ] Overlay firing rates of all 5 neurons alongside the trajectory
- [ ] Compare strategies qualitatively: do HP individuals exhibit different patterns of neural activity than non-HP?

### 7b. Per-neuron viable-range diagnostics across evolution

- [ ] For each generation, sample the best individual from the population, run a representative trial, record the fraction of timesteps each neuron's firing rate is in $[H_L, H_U]$
- [ ] Plot this fraction over generations, per condition
- [ ] Connect to substrate-level findings: does evolution itself push non-HP networks toward the viable range, or stay saturated?

### 7c. The frozen-HP test (the Stolting et al. test)

- [ ] For each evolved HP-during-behaviour individual, take the final network state (weights, biases as they are at the end of evolution)
- [ ] Re-run fitness evaluation with HP frozen at that point
- [ ] Measure fitness drop
- [ ] Compare with control: take HP-during-development individuals (where HP was already frozen for trials) and re-evaluate to estimate measurement variance
- [ ] Result interpretation: large fitness drop in HP-during-behaviour group → behaviour was relying on HP-enabled dynamics (Stolting et al.'s prediction); small drop → behaviour was task-driven and HP was incidental

**Gate:** three substantive analyses, each producing publishable-quality figures and a clear empirical claim.

---

## Phase 8 — Buffer / optional extension

If everything above completes ahead of schedule. Pick at most one.

- [ ] Run a small number of HP-during-behaviour individuals through an oscillation analysis: do their neural states oscillate in ways that stop when HP is frozen, paralleling Stolting et al.'s Figure 1?
- [ ] Include the discrimination task as a secondary replication
- [ ] Vary $\tau_w, \tau_b$ on HP-during-behaviour to test Stolting et al.'s timescale-separation findings

---

## Phase 9 — Analysis and writing

Order: methods → results → analyses → discussion → introduction → abstract.

### Methods (the highest-priority section)

- [ ] CTRNN equations and parameter ranges, every symbol defined
- [ ] HP equations and parameter values, every symbol defined
- [ ] Sensor model with the ray-intersection equation
- [ ] Agent body and motor model
- [ ] Falling shapes specification (positions, velocities, sizes, shape boundaries)
- [ ] Fitness function written out explicitly
- [ ] GA: pseudocode, encoding, population/generations, selection, mutation
- [ ] System diagram (refined from the Phase 1 sketch)
- [ ] **Symbol audit pass:** every variable in every equation must be defined on the same page; remove anything that appears once and is not used elsewhere

### Results

- [ ] Substrate-level sanity check (firing rate distributions before/after HP on random networks)
- [ ] Williams replication: four-condition fitness curves with error bands; final-fitness box plots
- [ ] Trajectory examples per condition with neural state alongside
- [ ] Per-neuron viable-range diagnostics across evolution
- [ ] Frozen-HP test: fitness-before-and-after-freezing per individual, per condition

### Analyses

- [ ] Qualitative comparison of evolved strategies across conditions
- [ ] Quantitative test: significance of the frozen-HP fitness drop
- [ ] Connection between substrate-level and evolutionary results via the viable-range diagnostics
- [ ] Failure-mode notes: do failing runs in HP-during-behaviour share a signature?

### Discussion

Pre-planned threads:

- [ ] Return to the working definition of adaptivity: which timescale of adaptation are we observing, and what is the regulating mechanism
- [ ] Substrate-level gains vs evolvability gains: Williams & Noble (2007) showed substrate gains; this project's results bear on the evolvability question by characterising the dynamics that actually evolve
- [ ] The Stolting et al. hypothesis: did our frozen-HP test support, complicate, or refute it?
- [ ] HP as cybernetic negative feedback at the neural level; Ashby's ultrastability mapping; the developmental-vs-online split as ontogenetic adaptation timescale
- [ ] Limitations: single task, single body, fixed $\tau_w$ and $\tau_b$, modest run counts
- [ ] Future work: discrimination task; alternative $\rho$ shapes; timescale-separation sweep

### Introduction (last, working backward from the discussion)

- [ ] Set up the substrate-vs-evolvability tension as the motivating question
- [ ] Introduce the working definition of adaptivity that the discussion returns to
- [ ] Wider context: HP in biological neural systems; evolutionary robotics tradition; cybernetic and dynamical-systems framings
- [ ] State the project's contribution clearly: replicate Williams Chapter 7 in our own simulator; add behavioural-trajectory, per-neuron-diagnostic, and frozen-HP analyses; test Stolting et al.'s hypothesis in Williams' setting

### Final polish

- [ ] All figures numbered and referenced from prose
- [ ] All equations numbered and referenced
- [ ] All citations in bibliography; no uncited entries; no unreferenced citations
- [ ] Bibliography ~10 entries
- [ ] LaTeX quote characters correct (`` '' or `\enquote{}`)
- [ ] No duplicated code in submission

---

## Reading queue

In project files:
- `williams2005homeostatic.pdf` — read
- `williams2007homeostatic.pdf` — read
- `Williams_Thesis.pdf` (Chapters 6 and 7) — read
- `Stolting_et_al___2023__Characterizing_the_Role_of_Homeostatic_Plasticity_in_Central_Pattern_Generators.pdf` — read
- `SAB96Beer.ps` — read (agent specification)

To track down during writing:
- Beer (1995) "On the dynamics of small CTRNNs" *Adaptive Behavior* — canonical CTRNN reference
- Mathayomchan & Beer (2002) — centre-crossing networks; for comparison with HP in discussion
- Di Paolo (2000) — adaptivity definition; antecedent to Williams' HP rule
- Ashby (1960) *Design for a Brain* — for the discussion framing
- Turrigiano (1999 or later) — biological HP background
- Funahashi & Nakamura (1993) — universal approximation result for CTRNNs (optional, for introduction)

---

## Reference snippets

### CTRNN state equation

$$\tau_y \dot{y}_i = -y_i + \sum_{j=1}^{N} w_{ji} z_j + I_i$$

### Activation function

$$z_i = \sigma(y_i + b_i) = \frac{1}{1 + e^{-(y_i + b_i)}}$$

### HP rule

$\rho(z) = (H_L - z)/H_L$ if $z < H_L$; $0$ if $H_L \le z \le H_U$; $(H_U - z)/(1 - H_U)$ if $z > H_U$.

$\tau_w \dot{w} = \rho |w|$, $\tau_b \dot{b} = \rho$.

**Williams Chapter 7 values**: $H_L = 0.2$, $H_U = 0.8$, $\tau_w = 40$, $\tau_b = 20$. Integration step $dt = 0.2$.

(The 0.25/0.75 values in earlier Williams chapters and in the conference paper apply to the substrate-level experiments, not the evolvability experiments. Use the Chapter 7 values for replication consistency.)

### Parameter ranges

$w \in [-10, 10]$, $b \in [-10, 10]$, $\tau_y \in [1, 4]$.

### Ray sensor

$S = S_{max}(D_{max} - D)/D_{max}$, with $S_{max} = 5$, $D_{max} = 100$. Three sensors in upward-facing fan spanning $\pi/6$ radians.

### Agent kinematics

$\tau_x \dot{x} = z_{right} - z_{left}$, $\tau_x = 0.2$. Agent radius 5.

### Environment notes

- Python with `uv` for environment management
- `numpy`, `scipy`, `matplotlib`, `pandas`, `tqdm` — all approved
- No Sandbox, no `madvn/CTRNN` for the final implementation (own CTRNN is justified for full methods description)
- Ubuntu, IntelliJ, bash terminal
