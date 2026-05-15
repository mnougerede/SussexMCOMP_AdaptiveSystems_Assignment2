# Assignment 2 — CTRNN Homeostatic Plasticity: To-Do List

**Submission deadline:** 28 May 2026.
**Today:** 15 May 2026. 13 days remain.

Two other assignments running in parallel. This is the live to-do list — re-check at the start of each working session.

---

## Project shape

**Headline:** Replicate Williams (2006) Chapter 7 ball-catching evolvability experiments. Extend with three analyses Williams did not perform: behavioural trajectory inspection, per-neuron viable-range diagnostics across evolution, and a frozen-HP test on HP-during-behaviour individuals (testing the Stolting, Beer & Izquierdo 2023 HP-enabled-oscillation hypothesis in Williams' setting).

**Why this is the project:** Williams found HP-during-behaviour evolved poorly. Stolting et al. (2023) speculated, but did not test on Williams' agent, that HP-during-behaviour evolves limit cycles in the joint network-and-parameter state space which collapse when HP is frozen. Our frozen-HP test is exactly that test in Williams' setting.

**Implementation choices (settled):**
- Pure-Python simulator, our own code.
- CTRNN integration via `madvn/CTRNN` (Candadai 2020), vendored into `src/ctrnn/_madvn.py` and described in full in `notes/methods_log.md` §2.5 — Chris explicitly permits this in the approved software list provided the algorithms are described as if our own.
- Ball-catching task only. Discrimination skipped (Williams himself reports HP-plastic networks struggle with it; risk without payoff).
- Williams' Chapter 7 values throughout, including $H_L = 0.2$, $H_U = 0.8$ (not the 0.25/0.75 from earlier chapters).
- Scale-down: 5 runs per condition (Williams used 10); 30 × 300 GA (Williams used 50 × 500). Documented in `notes/design_decisions.md`; methods section will state the cost in statistical resolution.

---

## Lessons from Assignment 1 feedback (plan these in, do not retrofit)

1. **Methods is the highest-priority section.** Every CTRNN and HP equation, every symbol defined on the page. Sensor model, environment parameters, GA, initialisation — all explicit. **The methods log (`notes/methods_log.md`) is being built section-by-section as we code; Phase 9 is a polish pass over accumulated content, not a from-scratch write.**
2. **System diagram** — components and couplings (CTRNN ↔ body ↔ environment, HP as a separate adaptive process). Sketch now.
3. **Behavioural examples are an explicit deliverable**, not a nice-to-have.
4. **Definition-of-adaptivity thread.** Cite in introduction (Di Paolo grounded in Ashby), apply to the experiment, return in discussion.
5. **Wider-context threads pre-planned.** HP as cybernetic negative feedback; ontogenetic vs lifetime adaptation; HP-enabled oscillations as a dynamical-systems framing.
6. **Lit review substantial and integrated**, not a separate block. ~10 entries in the bibliography.

---

## Phase 0 — Grounding (done)

- [x] Williams 2005 conference paper
- [x] Williams 2006 thesis Chapters 6 and 7
- [x] Williams & Noble 2007 (substrate-level follow-up)
- [x] Beer 1996 (the original ball-catching agent specification)
- [x] Stolting, Beer & Izquierdo 2023 (HP-enabled oscillations)
- [x] Confirm canonical equation forms; sign-check the plasticity rule

---

## Phase 1 — Project setup (done bar diagram)

- [x] GitHub repo `SussexMCOMP_AdaptiveSystems_Assignment2`
- [x] Clone locally, `bootstrap.sh`, planning documents in place
- [x] First commit and push
- [ ] **Sketch the system diagram on paper.** Components: agent body, ray sensors, falling shape, CTRNN, HP module, GA. Couplings: sensorimotor loop, parameter-update loop, GA selection loop. Save as `notes/system_diagram_sketch.png`. *(Do this before the HP pass — it forces decisions about HP's interface to the CTRNN.)*

---

## Phase 2 — Submit proposal (blocking; do first)

- [x] Re-read existing proposal draft and recognise its problems
- [ ] **Draft proposal content reflecting Option A**
- [ ] Transfer into `proposal_form.docx`
- [ ] Email Chris with a brief, honest cover note acknowledging lateness
- [ ] Continue Phase 4 work in parallel; do not block on Chris's reply, but adjust if he responds with concerns

---

## Phase 2a — Persistence infrastructure (done)

- [x] Pass 1: Config dataclasses — `CTRNNConfig`, `HPConfig`, `EnvConfig`, `AgentConfig`, `GAConfig`, `RunConfig`, `Condition` enum; JSON round-trip with typed reconstruction
- [x] Pass 2: IO layer — atomic writes, manifest (idempotent register + update), checkpoint with RNG sidecar, per-generation history, best-per-gen, `get_git_commit`
- [x] Pass 3: Stub experiment runner — `run_experiment` with git-mismatch guard, checkpoint-based resumption, `__main__` smoke-test block; all tests pass

---

## Phase 3 — CTRNN module (done)

- [x] Pass 1a: Vendor `madvn/CTRNN` as `src/ctrnn/_madvn.py`; verify under numpy 2.2.4 with README oscillator example.
- [x] Pass 1b: `CTRNNAgent` wrapper class with sensor-neurons-first convention enforced as tested invariant; `step`, `reset`, `motor_outputs`; 5 tests pass.
- [x] Pass 1c: Genotype-to-phenotype mapping (`src/ctrnn/genotype.py`); row-major weight layout; 11 tests pass.
- [x] Pass 1d: Convert sparse upstream weights to dense in the wrapper; methods log §2.5 tidied.

**Status:** Phase 3 CTRNN code is complete. Methods log §2.4 and §2.5 fully written. Tests: 29/29 passing.

---

## Phase 4 — Homeostatic plasticity (NEXT)

Settle these in chat before writing prompts:

- [ ] **HP integration order within a step.** When HP is active during a trial: sensor → `agent.step(I)` → new firing rates → bias update via $\tau_b \dot{b} = \rho$ → weight update via $\tau_w \dot{w} = \rho|w|$. Confirm against Williams §7.4.1.
- [ ] **State persistence across trial boundaries.** HP-shifted weights and biases at the end of the developmental phase are carried into the trial. Across trials within one fitness evaluation, do we reset to the genotype-encoded parameters or keep accumulating? Williams: reset per trial (developmental phase reruns from the genotype each time).
- [ ] **Frozen-HP semantics.** "Frozen" means $\dot{w} = \dot{b} = 0$, not "reset to genotype values". The HP class needs a clean enable/disable.

Then:

- [ ] HP module pass: `src/plasticity/hp.py` with `HP` class operating on a `CTRNNAgent`. Methods: `step(agent)` which reads `agent.z`, computes $\rho$ per neuron, applies eq. 5 and eq. 6 via Euler integration with $\Delta t = 0.2$. Disable/enable via a flag. Tests: $z = 0.1 \Rightarrow \rho > 0, \Delta b > 0, \Delta |w| > 0$; $z = 0.5 \Rightarrow \rho = 0$, no change; $z = 0.9 \Rightarrow$ symmetric to $z = 0.1$; boundary values $z = H_L, H_U$ give $\rho = 0$; inhibitory and excitatory weights both shrink under $\rho < 0$.
- [ ] Substrate-level sanity check pass: standalone script `scripts/substrate_check.py` generating 100 random 5-node CTRNNs, recording firing-rate distributions before and after 6000 HP steps with $I = 0$, saving a histogram figure to `figs/substrate_check.pdf`. **This figure goes in the report.**

**Gate:** HP demonstrably moves neurons out of saturation in random networks; figure ready for inclusion in results.

---

## Phase 5 — Simulator body (sensors, agent, shapes, environment)

Implementation order (revised): **shapes → ray sensors → agent body → trial runner → visualisation**. Ray sensors need shapes to intersect with; agent body needs sensors to provide inputs; trial runner needs all three.

- [ ] **Shapes pass.** `src/environment/shapes.py`. Circles only (radius 10). Drop from height 100, horizontal offset drawn from 10 evenly-spaced positions in $[x_{\text{agent}} - 25, x_{\text{agent}} + 25]$. Vertical velocity uniform on $[-0.5, -0.2]$. Tests: spawn distribution, motion update, exit-when-passed.
- [ ] **Ray sensors pass.** `src/agent/sensors.py`. Three rays in upward-facing fan spanning $\pi/6$ rad, mounted on agent periphery. $S = S_{\max}(D_{\max} - D)/D_{\max}$, $S_{\max} = 5$, $D_{\max} = 100$. Ray-circle intersection only (no diamonds — discrimination is out of scope). Tests: known geometry — ray directly at a circle returns expected $S$; ray missing returns $S = 0$; ray angles correct relative to agent's centre.
- [ ] **Agent body pass.** `src/agent/body.py`. Circular, radius 5, horizontal motion only. $\tau_x \dot{x} = z_{\text{right}} - z_{\text{left}}$, $\tau_x = 0.2$. Tests: zero motor differential gives zero velocity; equal-and-opposite differentials give expected steady-state velocity.
- [ ] **Trial runner pass.** `src/environment/trial.py`. Sequential drop of $N$ shapes per trial; per-timestep recording of agent position, neural state, shape position; produces a "trial record" object the fitness function consumes. Tests: deterministic with fixed seed; correct number of shapes; recorded array shapes match expected.
- [ ] **Visualisation pass.** Static trajectory plot (agent x, shape (x, y) over time) plus neural-state subplots. Used for sanity-checking evolved individuals and for the trajectory analysis figures.

**Gate:** can construct a hand-coded controller, drop shapes, and watch the agent move sensibly.

---

## Phase 6 — Genetic algorithm

Settle in chat before writing prompts:

- [ ] **GA flavour.** Microbial GA (Harvey 2009) or simple truncation-with-elitism? Microbial is dead simple, conceptually clean, and Chris has shown it in lectures. Decision pending.
- [ ] **Mutation specification.** Williams: Gaussian perturbation with reflection at boundaries. Mutation rate and magnitude to confirm.
- [ ] **Population scale.** Confirmed: 30 × 300 (down from Williams' 50 × 500), with cost stated in the methods section.

Then:

- [ ] GA pass: `src/ga/ga.py` implementing chosen flavour. Genotype is real-valued vector of length 35 in $[-1, 1]$. Asexual, mutation only. Seeded reproducibility (tested via `test_evolve.py`'s resumption test).
- [ ] Wire GA into the existing `run_experiment` stub (`src/experiments/evolve.py`); replace the stub's random-noise placeholder with real evolution. The persistence infrastructure (manifest, checkpoints, history) is already there — the GA just calls into it.
- [ ] Fitness function pass: `src/environment/fitness.py`. Williams 2006 eq. 7.3: $F = \sum_k (1 - d_k / d_{\max})$ over trials, where $d_k$ is horizontal distance between agent and shape centres at the moment the shape's lowest point reaches the top of the agent. Tests: known agent and shape positions give known $F$; out-of-range cases handled.
- [ ] Baseline check: evolve no-HP CTRNNs for a small number of runs (e.g. 30 × 100); confirm fitness curves rise from generation 0 and best individuals catch most balls.

**Gate:** evolution works on the no-HP baseline; checkpointing demonstrably resumes mid-run.

---

## Phase 7 — Williams replication (Milestone)

Four conditions:

1. **No HP** — random non-plastic CTRNN; HP off everywhere.
2. **HP during development only** — 6000 HP timesteps before each trial with $I = 0$ (or with environmental input — settle in HP pass); then frozen for the trial.
3. **HP during behaviour** — HP active throughout every trial; no separate developmental phase.
4. **HP during development and behaviour** — both.

- [ ] 5 runs per condition (scaled from Williams' 10)
- [ ] Best-fitness-per-generation curves with error bands across runs (Williams Fig. 7.2 equivalent)
- [ ] Final-fitness box plots across runs
- [ ] Qualitative check: does Williams' ordering hold? — developmental-only and no-HP roughly equivalent in final fitness, with developmental showing faster early progress; HP-during-behaviour worse

**Gate:** Williams' core qualitative result reproduced. This is the headline replication regardless of what comes next.

---

## Phase 8 — Three analyses Williams did not perform (Milestone — the contribution)

### 8a. Behavioural trajectory analysis

- [ ] Pick representative individuals from each condition (best-of-best, median-of-best, worst-of-best)
- [ ] Plot agent $x$ and shape $(x, y)$ over time across a small number of trials
- [ ] Overlay firing rates of all 5 neurons alongside the trajectory
- [ ] Compare qualitatively: do HP individuals exhibit different patterns of neural activity than non-HP?

### 8b. Per-neuron viable-range diagnostics across evolution

- [ ] For each generation, sample the best individual; run a representative trial; record per neuron the fraction of timesteps firing rate is in $[H_L, H_U]$
- [ ] Plot this fraction over generations, per condition
- [ ] Connect to substrate-level findings: does evolution itself push non-HP networks toward the viable range, or do they stay saturated?

### 8c. Frozen-HP test (the Stolting et al. test)

- [ ] Settle the control design first: HP-during-development individuals are not a clean variance control because their evolved weights differ from HP-during-behaviour individuals'. Better: re-evaluate each HP-during-behaviour individual N times with different shape sequences, compute within-individual fitness variance; that *is* the noise floor against which the freezing effect is measured.
- [ ] For each evolved HP-during-behaviour individual, take final $(w, b)$
- [ ] Re-run fitness evaluation with HP frozen at that point
- [ ] Measure fitness drop relative to the same individual's evaluation with HP active
- [ ] Test significance against the within-individual variance baseline
- [ ] Interpretation: large fitness drop → behaviour was relying on HP-enabled dynamics (Stolting et al.'s prediction); small drop → HP was incidental

**Gate:** three substantive analyses, each producing a publishable-quality figure and a clear empirical claim.

---

## Phase 9 — Buffer / optional extension

If everything above completes ahead of schedule. Pick at most one.

- [ ] Run a small number of HP-during-behaviour individuals through an oscillation analysis: do neural states oscillate in ways that stop when HP is frozen, paralleling Stolting et al.'s Fig. 1?
- [ ] Vary $\tau_w, \tau_b$ on HP-during-behaviour to test Stolting et al.'s timescale-separation findings
- [ ] Include the discrimination task as a secondary replication

---

## Phase 10 — Analysis and writing

Order: methods polish → results → analyses → discussion → introduction → abstract.

### Methods (the highest-priority section — but already largely written in `notes/methods_log.md`)

- [ ] Final pass over `notes/methods_log.md`: tidy prose, remove `[TBD]` markers (all sections should be filled by Phase 8 completion)
- [ ] System diagram (refined from Phase 1 sketch)
- [ ] **Symbol audit pass:** every variable in every equation must be defined on the same page; remove anything that appears once and is not used elsewhere

### Results

- [ ] Substrate-level sanity check figure (from Phase 4) — firing rate distributions before/after HP on random networks
- [ ] Williams replication: four-condition fitness curves with error bands; final-fitness box plots
- [ ] Trajectory examples per condition with neural state alongside (Phase 8a)
- [ ] Per-neuron viable-range diagnostics across evolution (Phase 8b)
- [ ] Frozen-HP test: fitness-before-and-after-freezing per individual, per condition (Phase 8c)

### Analyses

- [ ] Qualitative comparison of evolved strategies across conditions
- [ ] Quantitative test: significance of the frozen-HP fitness drop
- [ ] Connection between substrate-level and evolutionary results via the viable-range diagnostics
- [ ] Failure-mode notes: do failing runs in HP-during-behaviour share a signature?

### Discussion

Pre-planned threads:

- [ ] Return to the working definition of adaptivity: which timescale of adaptation are we observing, what is the regulating mechanism
- [ ] Substrate-level gains vs evolvability gains: Williams & Noble (2007) showed substrate gains; this project bears on the evolvability question by characterising the dynamics that actually evolve
- [ ] The Stolting et al. hypothesis: did our frozen-HP test support, complicate, or refute it?
- [ ] HP as cybernetic negative feedback at the neural level; Ashby's ultrastability mapping; developmental-vs-online split as ontogenetic adaptation timescale
- [ ] Limitations: single task, single body, fixed $\tau_w$ and $\tau_b$, modest run counts, scaled-down GA
- [ ] Future work: discrimination task; alternative $\rho$ shapes; timescale-separation sweep; larger networks (Williams Chapter 7 Experiment 4)

### Introduction (last, working backward from the discussion)

- [ ] Set up the substrate-vs-evolvability tension as the motivating question
- [ ] Introduce the working definition of adaptivity that the discussion returns to
- [ ] Wider context: HP in biological neural systems; evolutionary robotics tradition; cybernetic and dynamical-systems framings
- [ ] State the project's contribution: replicate Williams Chapter 7 in our own (vendored-CTRNN) simulator; three new analyses; test Stolting et al.'s hypothesis in Williams' setting

### Final polish

- [ ] All figures numbered and referenced from prose
- [ ] All equations numbered and referenced
- [ ] All citations in bibliography; no uncited entries; no unreferenced citations
- [ ] Bibliography ~10 entries
- [ ] LaTeX quote characters correct (`` '' or `\enquote{}`)
- [ ] No duplicated code in submission
- [ ] Submitted code is the `main` branch of the GitHub repo at submission time; a tag marks the submission commit

---

## Time budget (rough — 13 days, two other assignments in parallel)

- **Today (15 May):** Proposal draft + send to Chris. Phase 4 HP design discussion. System diagram sketch.
- **16–17 May:** HP module (Pass 4a). Substrate-level sanity check figure (Pass 4b).
- **18–20 May:** Phase 5 simulator body — shapes, sensors, agent body, trial runner.
- **21–22 May:** Phase 6 GA + fitness function + baseline check.
- **23–24 May:** Phase 7 Williams replication runs (overnight runs; multiple machines if possible).
- **25–26 May:** Phase 8 analyses. Figures from saved data.
- **27 May:** Writing polish, symbol audit, references.
- **28 May:** Submit by mid-afternoon. **Buffer of one half-day. Use it.**

Critical-path risks:
- Phase 7 evolutionary runs take longer than estimated. **Mitigation:** start them as soon as Phase 6 baseline passes; let them run in the background while writing.
- Phase 8c frozen-HP test reveals a bug in the HP module. **Mitigation:** the substrate-level sanity check (Phase 4) catches most HP bugs; Phase 6 baseline catches GA bugs; the only thing left to surface in Phase 8 is HP-trial-time bugs, which is a small surface area.
- Writing time underestimated. **Mitigation:** methods log is being written as we go; results sections are mostly figure captions plus a paragraph each; discussion threads are pre-planned in Phase 10.

---

## Reading queue

**In project files (read):**
- `williams2005homeostatic.pdf`, `williams2007homeostatic.pdf`, `Williams_Thesis.pdf` (Chs. 6 and 7), Stolting et al. 2023, Beer SAB96.

**To track down during writing:**
- Beer (1995) "On the dynamics of small CTRNNs" *Adaptive Behavior* — canonical CTRNN reference.
- Mathayomchan & Beer (2002) — centre-crossing networks; for comparison with HP in discussion.
- Di Paolo (2000) — adaptivity definition; antecedent to Williams' HP rule.
- Ashby (1960) *Design for a Brain* — discussion framing.
- Turrigiano (1999 or later) — biological HP background.
- Harvey (2009) microbial GA, if we go that route.
- Candadai (2020) `madvn/CTRNN` — software citation.

---

## Reference snippets

### CTRNN state equation

$$\tau_y \dot{y}_i = -y_i + \sum_{j=1}^{N} w_{ji} z_j + I_i$$

### Activation function

$$z_i = \sigma(y_i + b_i) = \frac{1}{1 + e^{-(y_i + b_i)}}$$

### HP rule

$\rho(z) = (H_L - z)/H_L$ if $z < H_L$; $0$ if $H_L \le z \le H_U$; $(H_U - z)/(1 - H_U)$ if $z > H_U$.

$\tau_w \dot{w} = \rho |w|$, $\tau_b \dot{b} = \rho$.

**Williams Chapter 7 values**: $H_L = 0.2$, $H_U = 0.8$, $\tau_w = 40$, $\tau_b = 20$. Integration step $\Delta t = 0.2$.

(The 0.25/0.75 values in earlier Williams chapters and in the conference paper apply to the substrate-level experiments, not the evolvability experiments. Use Chapter 7 values for replication consistency.)

### Parameter ranges

$w \in [-10, 10]$, $b \in [-10, 10]$, $\tau_y \in [1, 4]$.

### Ray sensor

$S = S_{\max}(D_{\max} - D)/D_{\max}$, with $S_{\max} = 5$, $D_{\max} = 100$. Three sensors in upward-facing fan spanning $\pi/6$ rad.

### Agent kinematics

$\tau_x \dot{x} = z_{\text{right}} - z_{\text{left}}$, $\tau_x = 0.2$. Agent radius 5.

### Environment notes

- Python with `uv` for environment management
- `numpy`, `scipy`, `matplotlib`, `pandas`, `tqdm` — all approved
- `madvn/CTRNN` vendored as `src/ctrnn/_madvn.py` (MIT licensed; see `LICENSE_THIRD_PARTY`). Described in `notes/methods_log.md` §2.5 as required by marking criteria.
- Ubuntu, IntelliJ, bash terminal.
