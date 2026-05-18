# Assignment 2 — CTRNN Homeostatic Plasticity: To-Do List

**Submission deadline:** 28 May 2026 (late period).
**Today:** 16 May 2026. 12 days remain.

Two other assignments running in parallel. This is the live to-do list — re-check at the start of each working session.

---

## Project shape

**Headline:** Replicate Williams (2006) Chapter 7 ball-catching evolvability experiments. Extend with three analyses Williams did not perform: behavioural trajectory inspection, per-neuron viable-range diagnostics across evolution, and a frozen-HP test on HP-during-behaviour individuals (testing whether genetic assimilation has occurred, and the Stolting et al. 2023 HP-enabled-oscillation hypothesis as a candidate mechanism).

**Framing (post Chris's feedback, 16 May):**
- Project framed primarily via the Baldwin effect: lifetime adaptation (HP) can guide evolution and lead to genetic assimilation. The frozen-HP test asks whether assimilation has occurred.
- The Stolting et al. hypothesis is a complementary *mechanism* by which assimilation can fail.
- HP does *not* create a "moving fitness landscape" (Chris flagged this in the proposal). What it changes is the genotype-phenotype mapping — search dynamics across a fixed landscape, not the landscape itself.

**Implementation choices (settled):**
- Pure-Python simulator, our own code.
- CTRNN integration via `madvn/CTRNN` (Candadai 2020), vendored into `src/ctrnn/_madvn.py` and described in full in `notes/methods_log.md` §2.5.
- Ball-catching task only. Discrimination skipped.
- Williams's Chapter 7 values throughout, including $H_L = 0.2$, $H_U = 0.8$.
- Developmental phase: zero input ($I = 0$), 6000 timesteps.
- **GA: tournament selection (K = 3) with elitism of 1, Gaussian mutation.** Not Williams's roulette + top-5 elitism. Chris's feedback (16 May) flagged Williams's combination as a fast route to premature convergence; tournament is the EC textbook default.
- GA scale: target Williams's 50 × 500 × 10; fallback 30 × 300 × 5 if the GA baseline check shows the full scale exceeds time budget.

---

## Lessons from Assignment 1 feedback (plan these in, do not retrofit)

1. **Methods is the highest-priority section.** Every CTRNN and HP equation, every symbol defined on the page. The methods log (`notes/methods_log.md`) is being built section-by-section as we code; Phase 9 is a polish pass over accumulated content, not a from-scratch write.
2. **System diagram** — sketch needed; Phase 1.
3. **Behavioural examples** are an explicit deliverable, not a nice-to-have.
4. **Definition-of-adaptivity thread** cited in introduction (Di Paolo grounded in Ashby), applied to the experiment, returned to in discussion.
5. **Baldwin-effect thread** added 16 May after Chris's feedback. Introduction → Methods (no change) → Discussion. References: Baldwin 1896, Hinton & Nowlan 1987, Mayley 1996, Turney 1996.
6. **Lit review substantial and integrated**, not a separate block.

---

## Phase 0 — Grounding (done)

- [x] Williams 2005 conference paper
- [x] Williams 2006 thesis Chapters 6 and 7
- [x] Williams & Noble 2007 (substrate-level follow-up)
- [x] Beer 1996 (the original ball-catching agent specification)
- [x] Stolting, Beer & Izquierdo 2023 (HP-enabled oscillations)

---

## Phase 1 — Project setup (mostly done)

- [x] GitHub repo `SussexMCOMP_AdaptiveSystems_Assignment2`
- [x] Clone locally, `bootstrap.sh`, planning documents in place
- [x] First commit and push
- [ ] **Sketch the system diagram on paper.** Components: agent body, ray sensors, falling shape, CTRNN, HP module, GA. Three coupled adaptive processes on different timescales. Save as `notes/system_diagram_sketch.png`. *(Do this before the HP pass.)*

---

## Phase 2 — Submit proposal (done)

- [x] Proposal drafted in LaTeX, rendered to PDF, sent to Chris on 15 May
- [x] Chris responded 16 May — positive overall ("excellent plan ... could be very good"); three substantive points addressed in updated `notes/design_decisions.md`:
  - "Moving fitness landscape" wording → corrected to search-dynamics framing
  - Baldwin effect → added as primary theoretical frame
  - Williams's GA → switched to tournament + elitism

---

## Phase 2a — Persistence infrastructure (done)

- [x] Pass 1: Config dataclasses
- [x] Pass 2: IO layer with atomic writes, manifest, checkpoint+RNG sidecar, history, best-per-gen
- [x] Pass 3: Stub experiment runner with git-mismatch guard and resumption

---

## Phase 3 — CTRNN module (done)

- [x] Pass 1a: Vendor `madvn/CTRNN` as `src/ctrnn/_madvn.py`; verify under numpy 2.2.4
- [x] Pass 1b: `CTRNNAgent` wrapper with sensor-neurons-first convention as tested invariant
- [x] Pass 1c: Genotype-to-phenotype mapping with row-major weight layout
- [x] Pass 1d: Dense weight storage in wrapper (upstream sparse converted); methods log §2.5 tidied

**Status:** 29/29 tests passing.

---

## Phase 4 — Homeostatic plasticity (NEXT)

Settle these in chat before writing prompts:

- [ ] **HP integration order within a step.** When HP is active during a trial: sensor → `agent.step(I)` → new firing rates → bias update via $\tau_b \dot b = \rho$ → weight update via $\tau_w \dot w = \rho|w|$. Confirm against Williams §7.4.1.
- [ ] **State persistence across trial boundaries.** Within one fitness evaluation (10 trials), HP-shifted weights and biases reset to genotype values between trials. Confirm: each trial in an evaluation runs the developmental phase from the genotype-encoded values, fresh.
- [ ] **Frozen-HP semantics.** "Frozen" means $\dot w = \dot b = 0$, not "reset to genotype values". HP class needs a clean enable/disable.

Then:

- [ ] **HP module pass** — `src/plasticity/hp.py` with `HP` class operating on a `CTRNNAgent`. Methods: `step(agent)` reads `agent.z`, computes $\rho$ per neuron, applies eq. 5 and eq. 6 via Euler integration with $\Delta t = 0.2$. Disable/enable via flag. Tests: $z = 0.1 \Rightarrow \rho > 0, \Delta b > 0, \Delta |w| > 0$; $z = 0.5 \Rightarrow \rho = 0$; $z = 0.9 \Rightarrow$ symmetric to $z = 0.1$; boundary values $z = H_L, H_U$ give $\rho = 0$; inhibitory and excitatory weights both shrink under $\rho < 0$.
- [ ] **Substrate-level sanity check** — standalone script `scripts/substrate_check.py` generating 100 random 5-node CTRNNs, recording firing-rate distributions before and after 6000 HP steps with $I = 0$, saving a histogram figure to `figs/substrate_check.pdf`. **This figure goes in the report.**

**Gate:** HP demonstrably moves neurons out of saturation in random networks; figure ready for inclusion in results.

---

## Phase 5 — Simulator body (sensors, agent, shapes, environment)

Order: **shapes → ray sensors → agent body → trial runner → visualisation.**

- [ ] **Shapes pass.** `src/environment/shapes.py`. Circles only (radius 10). Drop from height 100, horizontal offset and velocity per Williams Ch. 7. Tests: spawn distribution, motion update, exit-when-passed.
- [ ] **Ray sensors pass.** `src/agent/sensors.py`. Three rays, $\pi/6$ fan, upward-facing. $S = S_{\max}(D_{\max} - D)/D_{\max}$, $S_{\max} = 5$, $D_{\max} = 100$. Ray-circle intersection only. Tests: known geometry — ray directly at a circle returns expected $S$; ray missing returns $S = 0$.
- [ ] **Agent body pass.** `src/agent/body.py`. Circular, radius 5. $\tau_x \dot x = z_R - z_L$, $\tau_x = 0.2$. Tests: zero motor differential → zero velocity; equal-and-opposite → expected steady-state velocity.
- [ ] **Trial runner pass.** `src/environment/trial.py`. Sequential drop of $N$ shapes per trial; per-timestep recording of agent position, neural state, shape position; produces a "trial record" object the fitness function consumes. Tests: deterministic with fixed seed; correct number of shapes; recorded array shapes match expected.
- [ ] **Visualisation pass.** Static trajectory plot (agent x, shape (x, y) over time) plus neural-state subplots.

**Gate:** can construct a hand-coded controller, drop shapes, watch the agent move sensibly.

---

## Phase 6 — Genetic algorithm and fitness function

Settle in chat before writing prompts:

- [ ] **Mutation parameters** — start $p_m = 0.1$, $\sigma_m = 0.1$; tune in the GA baseline check.
- [ ] **Williams's eq. 7.3 — re-read carefully.** It is a *combined* displacement-reduction + final-distance score, not just final distance. Earlier drafts of methods log were wrong. Get it right before the fitness pass.

Then:

- [ ] **GA pass** — `src/ga/ga.py` implementing tournament selection (K = 3) + elitism of 1 + Gaussian mutation with boundary reflection. Genotype length 35, alleles in $[-1, 1]$. Seeded reproducibility (already tested via `test_evolve.py`'s resumption test).
- [ ] **Fitness function pass** — `src/environment/fitness.py`. Williams eq. 7.3 (full form). Tests: known agent and shape positions give known $F$; out-of-range cases handled.
- [ ] **Wire GA into existing `run_experiment` stub** (`src/experiments/evolve.py`); replace stub's random-noise placeholder with real evolution. Persistence already there.
- [ ] **Baseline check.** Evolve no-HP CTRNNs for a small number of runs (e.g. 30 × 100); confirm fitness curves rise above the random baseline; best evolved individuals catch most balls in visual trajectory inspection; measure runtime per evaluation. **Use this measurement to decide: Williams-scale or scaled-down for Phase 7.**

**Gate:** evolution works on the no-HP baseline; checkpointing demonstrably resumes mid-run; runtime measured.

---

## Phase 7 — Williams replication (Milestone)

Four conditions (per `notes/methods_log.md` §9):

1. No HP
2. HP during development only
3. HP during behaviour only, no developmental phase
4. HP during development and behaviour

- [ ] 5 or 10 runs per condition (decision after Phase 6 runtime check)
- [ ] Best-fitness-per-generation curves with error bands across runs (Williams Fig. 7.2 equivalent)
- [ ] Final-fitness box plots across runs
- [ ] Qualitative check: does Williams's ordering hold *with our better-conditioned GA*? — note that if it doesn't, this is itself a substantive finding

**Gate:** four-condition fitness plot ready for the report.

---

## Phase 8 — Three analyses Williams did not perform (Milestone — the contribution)

### 8a. Behavioural trajectory analysis

- [ ] Pick representative individuals per condition (best-of-best, median-of-best, worst-of-best)
- [ ] Plot agent $x$ and shape $(x, y)$ over time across a small number of trials, neural state alongside
- [ ] Compare qualitatively across conditions; connect to per-neuron diagnostics

### 8b. Per-neuron viable-range diagnostics across evolution

- [ ] For each generation, sample the best individual; replay a representative trial; record per neuron the fraction of timesteps firing rate in $[H_L, H_U]$
- [ ] Plot fraction over generations, per condition
- [ ] **Interpret via Baldwin frame:** does the in-range fraction in non-HP networks rise over generations (evolution discovering what HP would have enforced)? Does the HP-during-behaviour viable-range fraction depend on HP being active?

### 8c. Frozen-HP test (assimilation test)

- [ ] **Control design.** Re-evaluate each HP-during-behaviour individual N times with different shape sequences to estimate within-individual fitness variance. This is the noise floor.
- [ ] For each evolved HP-during-behaviour individual, take final $(w, b)$
- [ ] Re-run fitness evaluation with HP frozen at that point
- [ ] Measure fitness drop relative to the same individual's evaluation with HP active
- [ ] Test significance against the within-individual variance baseline
- [ ] **Interpretation (Baldwin frame):** large drop → genetic assimilation has *not* occurred. Small drop → assimilation occurred. **(Stolting frame):** large drop is consistent with HP-enabled limit cycles being essential to behaviour; small drop is inconsistent with this mechanism.

**Gate:** three substantive analyses, each producing a publishable-quality figure and a clear empirical claim.

---

## Phase 9 — Buffer / optional extension

If everything above completes ahead of schedule. Pick at most one.

- [ ] Oscillation analysis on HP-during-behaviour individuals (Stolting-style)
- [ ] Vary $\tau_w, \tau_b$ on HP-during-behaviour (Stolting timescale-separation test)
- [ ] Run a comparison with Williams's exact GA to isolate GA effects from HP effects

---

## Phase 10 — Analysis and writing

Order: methods polish → results → analyses → discussion → introduction → abstract.

### Methods (already largely written in `notes/methods_log.md`)

- [ ] Final pass over `notes/methods_log.md`: tidy prose, remove `[TBD]` markers
- [ ] System diagram (refined from Phase 1 sketch)
- [ ] Symbol audit: every variable defined on the same page; remove unused definitions

### Results

- [ ] Substrate-level sanity check figure (from Phase 4)
- [ ] Williams replication: four-condition fitness curves with error bands; final-fitness box plots
- [ ] Trajectory examples per condition with neural state alongside (Phase 8a)
- [ ] Per-neuron viable-range diagnostics across evolution (Phase 8b)
- [ ] Frozen-HP test: fitness-before-and-after-freezing per individual, per condition (Phase 8c)

### Analyses

- [ ] Qualitative comparison of evolved strategies across conditions
- [ ] Quantitative test: significance of the frozen-HP fitness drop
- [ ] Connection between substrate-level and evolutionary results via the viable-range diagnostics
- [ ] Failure-mode notes within HP-during-behaviour

### Discussion (pre-planned threads)

- [ ] Adaptivity definition revisited: which timescales of adaptation are observed
- [ ] **Baldwin effect / genetic assimilation as the central frame.** Citations: Baldwin 1896, Hinton & Nowlan 1987, Mayley 1996, Turney 1996.
- [ ] **Stolting et al. hypothesis as candidate mechanism** for failure of assimilation in HP-during-behaviour
- [ ] Search dynamics framing (not "moving landscape"): HP changes the genotype-phenotype mapping, smoothing or roughening search across a fixed landscape
- [ ] Substrate-level gains vs evolvability gains: how this work connects the two
- [ ] HP as cybernetic negative feedback at the neural level; Ashby ultrastability mapping
- [ ] GA choice: why we used tournament rather than Williams's GA, and what this means for the comparison
- [ ] Limitations: single task, fixed $\tau_w, \tau_b$, modest run counts, no discrimination
- [ ] Future work: discrimination task; timescale-separation sweep; alternative $\rho$ shapes; GA comparison; larger networks

### Introduction (last)

- [ ] Working definition of adaptivity (Di Paolo, Ashby)
- [ ] HP background — biological (Turrigiano) and computational (Williams & Noble)
- [ ] Substrate-vs-evolvability tension as motivating question
- [ ] **Baldwin-effect framing** with key citations
- [ ] State the project's contribution: replication + three new analyses + frozen-HP/assimilation test

### Final polish

- [ ] All figures numbered and referenced from prose
- [ ] All equations numbered and referenced
- [ ] All citations in bibliography; no uncited entries; no unreferenced citations
- [ ] LaTeX quote characters correct
- [ ] Submitted code is the `main` branch of the GitHub repo at submission time; tag marks the submission commit

---

## Time budget (12 days, two other assignments in parallel)

- **Today (16 May):** Update planning docs (this pass). System diagram sketch. Settle HP integration-order and state-persistence questions.
- **17–18 May:** HP module (Pass 4a). Substrate-level sanity check figure.
- **19–21 May:** Phase 5 simulator body — shapes, sensors, agent body, trial runner.
- **22–23 May:** Phase 6 GA + fitness function + baseline check (incl. runtime measurement).
- **24–25 May:** Phase 7 Williams replication runs (overnight + multiple machines if possible).
- **26–27 May:** Phase 8 analyses; figures from saved data; writing.
- **28 May:** Submit by mid-afternoon. Half-day buffer.

Critical-path risks:
- Phase 7 evolutionary runs take longer than estimated. **Mitigation:** start them as soon as Phase 6 baseline passes; let them run in the background while writing.
- Phase 8c frozen-HP test reveals an HP bug. **Mitigation:** substrate-level sanity check (Phase 4) catches most HP bugs; Phase 6 baseline catches GA bugs.
- Writing time underestimated. **Mitigation:** methods log being written as we go; results sections are mostly figure captions + a paragraph each; discussion threads pre-planned in Phase 10.

---

## Reading queue

**In project files (read):**
- `williams2005homeostatic.pdf`, `williams2007homeostatic.pdf`, `Williams_Thesis.pdf` (Chs. 6 and 7), Stolting et al. 2023, Beer SAB96.

**To track down during writing:**
- Beer (1995) "On the dynamics of small CTRNNs" *Adaptive Behavior* — canonical CTRNN reference.
- Di Paolo (2000) — adaptivity definition.
- Ashby (1960) *Design for a Brain* — discussion framing.
- Turrigiano (1999 or later) — biological HP background.
- **Baldwin (1896) — Baldwin effect.**
- **Hinton & Nowlan (1987) — Baldwin in EC.**
- **Mayley (1996) — genetic assimilation in EC.**
- **Turney (1996) — myths of the Baldwin effect.**
- Candadai (2020) `madvn/CTRNN` — software citation.

---

## Reference snippets

### CTRNN state equation

$$\tau_y \dot{y}_i = -y_i + \sum_{j=1}^{N} w_{ji} z_j + I_i$$

### Activation function

$$z_i = \sigma(y_i + b_i) = \frac{1}{1 + e^{-(y_i + b_i)}}$$

### HP rule

$\rho(z) = (H_L - z)/H_L$ if $z < H_L$; $0$ if $H_L \le z \le H_U$; $(H_U - z)/(1 - H_U)$ if $z > H_U$.

$\tau_w \dot w = \rho |w|$, $\tau_b \dot b = \rho$.

**Williams Chapter 7 values**: $H_L = 0.2$, $H_U = 0.8$, $\tau_w = 40$, $\tau_b = 20$. Integration step $\Delta t = 0.2$.

### Parameter ranges

$w \in [-10, 10]$, $b \in [-10, 10]$, $\tau_y \in [1, 4]$.

### Ray sensor

$S = S_{\max}(D_{\max} - D)/D_{\max}$, with $S_{\max} = 5$, $D_{\max} = 100$. Three sensors in upward-facing fan spanning $\pi/6$ rad.

### Agent kinematics

$\tau_x \dot x = z_R - z_L$, $\tau_x = 0.2$. Agent radius 5.

### GA spec

Tournament selection $K = 3$, elitism of 1, no crossover, Gaussian mutation per allele $\mathcal{N}(0, \sigma_m^2)$ with reflection at $[-1, 1]$, $p_m$ and $\sigma_m$ tuned at GA baseline check.

### Environment notes

- Python with `uv` for environment management
- `numpy`, `scipy`, `matplotlib`, `pandas`, `tqdm` — all approved
- `madvn/CTRNN` vendored as `src/ctrnn/_madvn.py` (MIT licensed). Described in `notes/methods_log.md` §2.5.
- Ubuntu, IntelliJ, bash terminal.
