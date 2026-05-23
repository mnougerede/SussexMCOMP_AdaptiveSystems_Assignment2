# Assignment 2 — CTRNN Homeostatic Plasticity: To-Do List

**Submission deadline:** 28 May 2026 (late period).
**Today:** 16 May 2026. 12 days remain.

Two other assignments running in parallel. This is the live to-do list — re-check at the start of each working session.

---

## Project shape

**Headline:** Replicate Williams (2006) Chapter 7 ball-catching evolvability experiments. Extend with analyses Williams did not perform: three of the evolved controller (behavioural trajectory inspection, per-neuron viable-range diagnostics across evolution, and a frozen-HP test on HP-during-behaviour individuals testing whether genetic assimilation has occurred, with the Stolting et al. 2023 HP-enabled-oscillation hypothesis as a candidate mechanism) and one of the search itself (population-level search-dynamics analysis, the larger adaptive system the assignment requires an EA project to examine).

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

## Phase 4 — Homeostatic plasticity (done)

- [x] HP integration order confirmed: sensor → `agent.step(I)` → compute $\rho$ from new $z$ → bias update → weight update
- [x] State persistence confirmed: weights, biases, and neural states all reset to genotype values at start of each trial
- [x] Frozen-HP semantics confirmed: `enabled=False` is a no-op; does not reset parameters
- [x] **HP module** — `src/plasticity/hp.py`, `HP` class, 9/9 tests passing
- [x] **Substrate-level sanity check** — `scripts/substrate_check.py`, figure at `figs/substrate_check.pdf`; 86.1% → 47.7% outside $[H_L, H_U]$ during HP; 53.9% after HP off

**Status:** 38/38 tests passing (29 from Phase 3 + 9 new).

---

## Phase 5 — Simulator body (done)

- [x] **Shapes** — `src/environment/shapes.py`, 6/6 tests passing
- [x] **Ray sensors** — `src/agent/sensors.py`, 4/4 tests passing
- [x] **Agent body** — `src/agent/body.py`, 4/4 tests passing
- [x] **Trial runner** — `src/environment/trial.py` + `TrialRecord` dataclass, 6/6 tests passing; `CTRNNAgent` extended with `genotype` attribute and `load_genotype()` method
- [ ] **Visualisation pass** — static trajectory plot (agent x, shape (x,y) over time) plus neural-state subplots *(deferred — will do alongside Phase 8 analyses)*

**Status:** 58/58 tests passing across all phases to date.

**Gate:** ✓ trial runner deterministic with fixed seed; correct shape count; HP modes all verified.

---

## Phase 6 — Genetic algorithm and fitness function (done)

- [x] **Mutation parameters** — settled: $p_m = 0.03$ (Williams's rate), $\sigma_m = 0.1$, Gaussian with boundary reflection. Pure Gaussian preferred over Williams's mixed scheme; rate matches Williams.
- [x] **Williams eq. 7.3** — confirmed full form: combined displacement-reduction + final-distance score, per-shape $S_{\max} = (1 + |v_{x,\text{shape}}|) \cdot 100/|v_{y,\text{shape}}|$, $\phi$ clips negative reduction to 0.
- [x] **GA pass** — `src/ga/ga.py`, tournament $K=3$, elitism 1, Gaussian mutation, boundary reflection. 8/8 tests passing.
- [x] **Fitness function pass** — `src/environment/fitness.py`, Williams eq. 7.3 full form, $S_0 = 0$ handled gracefully. 11/11 tests passing.
- [x] **Wire GA into evolve.py** — real evolution replacing stub; `hp_mode` from condition enum; checkpointing every 10 generations. 63/63 tests passing.
- [x] **Baseline check script** — `scripts/ga_baseline_check.py`; fitness curve and trajectory figure saved to `figs/`.

**Status:** 63/63 tests passing.

---

## Phase 7 — Performance, infrastructure, and Williams replication (NEXT)

### 7a–7c: Performance and experiment infrastructure (done)

- [x] **Performance optimisations** — ray angles precomputed as module-level constants; CTRNN outputs setter bypassed in `euler_step`. Per-timestep cost: 79 µs → 31 µs (2.6× speedup). Full-experiment estimate: ~6.7h single-threaded.
- [x] **Multiprocessing** — `n_workers` field in `RunConfig`; `evolve.py` parallelises fitness evaluation across individuals via `multiprocessing.Pool`. Deterministic with fixed seed. Existing 63 tests pass.
- [x] **Experiment status and batch launcher** — `scripts/experiment_status.py` (read-only status view over all manifests); `scripts/launch_batch.py` (CLI launcher with provenance tracking in `batches/`); `plan/experiment_targets.json` (human-editable targets: 5 runs/condition, 200 generations).
- [x] **Backup plan** — `experiments/` gitignored (runtime output); backed up to OneDrive after each batch. `batches/` versioned as lightweight provenance.

### 7d: GA baseline check and scale decision (done)

- [x] **WSL2 setup on desktop** (i5-9600K, 6 cores) — done
- [x] **GA baseline check** — `no_hp`, 100 generations, pop 30, seed 42, `n_workers=6`. Mean fitness > 0.5 by gen 3, plateaued ~0.7–0.8. Best individual 0.85, trajectory shows clean shape tracking. ~17s/generation with 6 workers.
- [x] **Scale decision** — pop 30, 200 generations, 5 runs/condition, 3 trials/evaluation. (Population kept at 30 rather than 20 for richer dynamics; split across two machines to fit the time budget.) Note: HP conditions run slower than the `no_hp` baseline check predicted, due to the 6000-timestep developmental phase the baseline never measured.

### 7e: Williams replication runs (done — data collected)

Four conditions: `no_hp`, `dev_only`, `behaviour_only`, `both`. 5 runs each, 20 total. Split across two machines: 4 runs/condition on desktop (`replication_desktop`, seeds 100–403), 1 run/condition on laptop (`replication_laptop`, seeds 500–800). All 20 complete, 0 failures.

**Final best fitnesses (5 runs per condition, computed):**
- `no_hp`: 0.506, 0.812, 0.666, 0.674, 0.696 → mean 0.671, SD 0.109
- `dev_only`: 0.833, 0.838, 0.875, 0.728, 0.884 → mean 0.832, SD 0.062
- `behaviour_only`: 0.707, 0.674, 0.849, 0.684, 0.666 → mean 0.716, SD 0.076
- `both`: 0.568, 0.589, 0.653, 0.685, 0.821 → mean 0.663, SD 0.100

**Computed result (replication_figure.py, first batch n=5):** `dev_only` clearly strongest. Kruskal-Wallis across conditions H=9.19, p=0.027 (significant). Pairwise Mann-Whitney with Bonferroni (×6): the two interesting pairs, `no_hp` vs `dev_only` and `dev_only` vs `both`, both reach U=1, raw p=0.016, p_bonf=0.095 — the strongest separation n=5 can plausibly produce, still just outside the corrected 0.05 threshold. The n=5 two-sided resolution floor is p=0.0040 (U=0). `no_hp`, `behaviour_only`, `both` are mutually indistinguishable. Ordering matches Williams qualitatively except that online-HP conditions are NOT significantly worse than `no_hp` (Williams found them worse) — candidate discussion point: the detrimental online-HP effect may have been partly a GA-conditioning artefact in Williams's roulette+top-5 setup.

**Top-up decision:** running a second batch (`replication_desktop_extra`, base_seed 104, 5 more runs/condition) on the idle desktop to reach n=10/condition. Resume-tolerant; cost is near zero. Not because the current data is insufficient for the headline (it is sufficient — `dev_only` wins, the other three overlap), but because n=10 cleans up the three-way overlap story and matches Williams's runs-per-condition. Re-run replication_figure.py and search_dynamics.py when it completes; both pick up all runs automatically via the loader.

**Done (the analysis):**
- [x] Best-fitness-per-generation curves, mean with $\pm$1 SD band, individual runs faint — `figs/replication_fitness_curves.pdf`
- [x] Final-fitness box plots, 5 run points overlaid, no error band — `figs/replication_final_box.pdf`
- [x] Significance test (Kruskal-Wallis + pairwise Mann-Whitney with Bonferroni and the n=5 floor) — printed summary from replication_figure.py
- [x] Qualitative check: Williams's ordering holds qualitatively; online-HP-not-worse divergence noted above
- [x] **Replicate Williams's search observations:** the `dev_only` quicker-early-progress pattern is visible in the search-dynamics figure (steep rise in first ~5 generations); recorded in methods_log §11.4
- [ ] Figure polish (pending, bundled into one pass after all four analysis scripts exist): zoom y-axes on box plot and curves; show individual runs on the curves; CSV export of per-generation per-condition values

**Note:** the integrity assertion originally added to replication_figure.py (last-gen best == max-over-gens) was removed — it fired for all 20 runs because `best_fitness` in `history/` is the score evaluated that generation under stochastic re-evaluation, so the elitist genotype's *score* can vary even though the genotype is preserved. Elitism is on the genotype, not the re-evaluated score. Not a data problem.

---

## Phase 8 — Analyses Williams did not perform (Milestone — the contribution)

Three analyses of what evolved (8a trajectories, 8b viable-range, 8c frozen-HP) and one of how it evolved (8d search dynamics).

### 8a. Behavioural trajectory analysis

- [ ] Pick representative individuals per condition (best-of-best, median-of-best, worst-of-best)
- [ ] Plot agent $x$ and shape $(x, y)$ over time across a small number of trials, neural state alongside
- [ ] Compare qualitatively across conditions; connect to per-neuron diagnostics

### 8b. Per-neuron viable-range diagnostics across evolution

- [ ] For each generation, sample the best individual; replay a representative trial; record per neuron the fraction of timesteps firing rate in $[H_L, H_U]$
- [ ] Plot fraction over generations, per condition
- [ ] **Interpret via Baldwin frame:** does the in-range fraction in non-HP networks rise over generations (evolution discovering what HP would have enforced)? Does the HP-during-behaviour viable-range fraction depend on HP being active?

### 8c. Frozen-HP test (assimilation test)

Single variant: freeze established dynamics after settling (Stolting-faithful adiabatic elimination), not freeze-from-genotype. Genotype-freeze variant demoted to Phase 9 optional; the genotype/innate-competence reading comes from the cross-condition comparison instead.

- [ ] **Freeze semantics:** set $\dot w = \dot b = 0$ from the freeze point, holding $(w,b)$ at HP-driven values. Do NOT reset to genotype. Williams's term: adiabatic elimination.
- [ ] **Freeze point.** `both`: run developmental phase, run into trial, freeze at end of developmental phase (primary). `behaviour_only`: no developmental phase, so allow a settling window into the trial then freeze; state window length, confirm not knife-edge sensitive.
- [ ] **Within-individual variance baseline.** Re-evaluate each individual N times with different shape-sequence seeds, HP active, to estimate fitness noise. The drop is significant only if it exceeds this.
- [ ] **Controls (noise floor).** Re-evaluate `dev_only` and `no_hp` individuals under the identical freeze-and-continue procedure. Both behaved with HP off already, so the drop must be ~0. A non-trivial drop in either is a BUG signal, checked before interpreting any `behaviour_only`/`both` drop.
- [ ] For each `behaviour_only` and `both` individual: fitness with HP active vs fitness after freezing. Measure the drop.
- [ ] **Interpretation (Baldwin frame):** large drop → genetic assimilation has *not* occurred, behaviour is HP-dependent. Small drop → assimilation occurred. **(Stolting frame):** large drop is consistent with HP-enabled dynamics being essential to behaviour; small drop is inconsistent with this mechanism.

### 8d. Search-dynamics analysis (analysis of the larger adaptive system) — DONE (first batch)

Analysis of the search, not the evolved product. All from saved `history/` data (full population fitness array per generation), no re-running. Serves the assignment requirement that an EA project analyse the larger adaptive system. The replication half (early-progress, consistency) lives in Phase 7 results; this is the extension half. Script: `scripts/analysis/search_dynamics.py`. Figure: `figs/search_dynamics_population.pdf` (2x2, one panel per condition, shared y-axis; population best, population mean, fitness spread per generation).

- [x] Population fitness distribution over generations, per condition (best, mean, spread)
- [x] Diversity measure over generations: population fitness spread (per-generation SD of the 30 fitnesses). NOTE: genotype-based diversity is NOT computable — `history/` saves population fitnesses but not population genotypes. Fitness spread is the deliberate proxy; documented in the script and methods.
- [x] Convergence comparison: `dev_only` jumps in first ~5 gens then plateaus; others rise gradually over 200 gens; spread collapses to ~0 by gen 10–20 in all conditions
- [x] **Interpret via search-dynamics framing:** conditions differ in *where* they converge, not whether/how fast. Dev only converges higher. Full interpretation and the Both puzzle in methods_log §11.4 (flagged hypothesis pending 8b/8c)
- [ ] Re-run when `replication_desktop_extra` completes (n=10) to confirm patterns hold

**Loader + replication figure infrastructure (DONE):**
- [x] `scripts/analysis/load_runs.py` — shared loader, run discovery, condition resolution (config.json authoritative, manifest disagreement warns), npz reading, grouping. 16 synthetic-fixture tests, 79 total passing. Fix: path normalisation for the cross-machine `_resolve_output_dir` rglob fallback (desktop runs store `/home/mnoug/...` paths; loader patches `experiment_status.EXPERIMENTS_DIR` via realpath — comment in code, do not remove).
- [x] `scripts/analysis/replication_figure.py` — two PDFs + stats summary (see Phase 7 result above).

**Gate:** four substantive analyses, each producing a publishable-quality figure and a clear empirical claim.

---

## Phase 9 — Buffer / optional extension

If everything above completes ahead of schedule. Pick at most one.

- [ ] Oscillation analysis on HP-during-behaviour individuals (Stolting-style)
- [ ] Genotype-freeze variant of the frozen-HP test (freeze from genotype, HP never acts; the innate-competence reading). Only if we find we have too little to discuss; the cross-condition comparison already carries most of this signal.
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
- [ ] Williams replication: four-condition fitness curves (mean $\pm$1 SD, individual runs faint); final-fitness box plots with run points overlaid, no error band
- [ ] Search replication: early-progress and run-to-run consistency read off the replication curves (Phase 7)
- [ ] Trajectory examples per condition with neural state alongside (Phase 8a)
- [ ] Per-neuron viable-range diagnostics across evolution (Phase 8b)
- [ ] Frozen-HP test: fitness-before-and-after-freezing per individual, per condition (Phase 8c)
- [ ] Search-dynamics: population fitness distribution and diversity over generations, per condition (Phase 8d)

### Analyses

- [ ] Qualitative comparison of evolved strategies across conditions
- [ ] Quantitative test: significance of the frozen-HP fitness drop
- [ ] Connection between substrate-level and evolutionary results via the viable-range diagnostics
- [ ] Failure-mode notes within HP-during-behaviour
- [ ] Search-dynamics analysis: how HP and the GA shape the search across a fixed landscape (the larger adaptive system the assignment asks for)

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
- [ ] State the project's contribution: replication + four new analyses (three of the evolved controller, one of the search) including the frozen-HP/assimilation test

### Final polish

- [ ] All figures numbered and referenced from prose
- [ ] All equations numbered and referenced
- [ ] All citations in bibliography; no uncited entries; no unreferenced citations
- [ ] LaTeX quote characters correct
- [ ] Submitted code is the `main` branch of the GitHub repo at submission time; tag marks the submission commit

---

## Time budget (9 days remain, two other assignments in parallel)

- **19 May (today):** Doc updates. WSL2 setup on desktop. GA baseline check.
- **20–21 May:** Launch full replication runs on desktop overnight. Monitor with `experiment_status.py`.
- **22–23 May:** Phase 8 analyses from saved data; visualisation pass.
- **24–25 May:** Writing — methods polish, results, analyses.
- **26–27 May:** Writing — discussion, introduction, abstract.
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
