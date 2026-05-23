# Design decisions

A working document for the methodological choices in the project. Each section states the question, gives the current choice with rationale, and notes anything still open.

The project: replicate Williams (2006) Chapter 7 ball-catching evolvability experiments, plus four new analyses (behavioural trajectories, per-neuron viable-range diagnostics across evolution, frozen-HP test, and a population-level search-dynamics analysis).

---

## CTRNN implementation: vendor `madvn/CTRNN`

**Decision:** Use the `madvn/CTRNN` package (Candadai 2020), vendored as `src/ctrnn/_madvn.py` rather than installed from PyPI. Wrap it in a thin `CTRNNAgent` class that establishes the sensor/motor neuron indexing convention.

**Reasoning:** On the approved software list and used in Chris's own demos. Pure-Python, small, MIT licensed. Vendoring (rather than installing) keeps the source as part of the submission, makes it easy to read alongside the report, and removes dependency drift risk on an unmaintained library. The marking criteria require us to describe its algorithms as if our own; that price is paid by the methods section regardless of whether the implementation is ours or vendored, so the small-but-real numerical-correctness risk of a from-scratch reimplementation is uncompensated. See `notes/methods_log.md` §2.5 for the methods description.

**Trade-off explicitly considered:** writing our own CTRNN would have given more "ownership" of the numerical core, but the integration loop, sigmoid, gain handling, and state arrays are exactly the kind of code where small bugs are subtle and expensive to find. Vendoring a tested implementation is the lower-risk, lower-effort choice.

---

## Task: ball-catching only, not discrimination

**Decision:** Implement and run only the ball-catching task.

**Reasoning:** Williams himself reports in Chapter 7 that the discrimination task was very difficult for HP-plastic networks; performance remained poor across most conditions. Including it adds substantial implementation complexity (diamond geometry, two-shape trial generation, the discrimination fitness function) without a clear payoff: even if our results match Williams' patterns there, the patterns are weaker and harder to interpret.

**Mention in discussion:** the choice to use ball-catching only, with a note that discrimination is more demanding and was set aside for time-budget reasons. Frame discrimination as natural future work, with a prediction: if HP-during-behaviour is HP-dependent (per the frozen-HP test), the dependence should be even stronger on the harder discrimination task.

---

## CTRNN architecture: Williams Chapter 7 specification exactly

**Decision:** 5-node fully connected CTRNN, 3 sensor neurons and 2 motor neurons, **no interneurons**. Each ray sensor feeds a unique node. Motor output is read from the two non-sensor nodes.

**Reasoning:** Williams' Chapter 7 specification (page 120 of the thesis). Deviating from it complicates direct comparison with Williams' results. The absence of interneurons is conceptually striking but it's what Williams used. Mention in methods that this is a minimal architecture and that one could investigate larger networks (which Williams himself does in Chapter 7 Experiment 4).

---

## HP target range: Williams Chapter 7 values

**Decision:** $H_L = 0.2$, $H_U = 0.8$ throughout, including in the substrate-level sanity check.

**Reasoning:** Williams uses $[0.25, 0.75]$ for the substrate-level analyses in Chapter 6 and Williams & Noble (2007), but **switches to $[0.2, 0.8]$ in Chapter 7** (page 121 of the thesis). The values used should match the chapter being replicated, not the values quoted in lecture slides or the conference paper. Using one consistent pair throughout the project (including the sanity check) avoids the awkward case where the sanity check uses different values from the main experiments.

**Mention in methods:** flag this divergence from Chapter 6 / Williams & Noble explicitly.

---

## Plasticity timescales: Williams values

**Decision:** $\tau_w = 40$, $\tau_b = 20$.

**Reasoning:** Williams reports that both rules give similar results when applied independently. Combined rules are what he uses in the published comparisons, and these timescale values are reported as canonical. Sensitivity to these is mentioned in Williams' Chapter 6 results section but not systematically explored.

---

## Sensory input during developmental phase: zero input

**Decision:** During the 6000-timestep developmental phase, the network receives zero external input ($I = 0$). HP runs; the network state evolves.

**Reasoning:** Williams' thesis does not explicitly specify the input regime during the developmental phase — it says only "updating every network for 6000 timesteps before each trial began" (Chapter 7, §7.5.2). Three readings are defensible:

1. **Zero input.** Matches Williams's Chapter 6 substrate-level analyses, where HP is run on networks in isolation.
2. **Sensory input from a sample scene.** The network "watches" shapes fall without acting.
3. **Random input.** Approximating the eventual input distribution without committing to specific trials.

We choose (1). The Chapter 6 precedent is the strongest tie-break: Williams himself uses zero-input HP to demonstrate the substrate-level effect, and "pre-conditioning the network" is the operation he's most likely importing here. A 6000-timestep developmental phase is 1200 simulation seconds (at $\Delta t = 0.2$), which is much longer than a typical trial and supports a "settle the network's intrinsic properties" reading rather than a "show it the task" reading.

**Mention in methods:** state this is a documented choice; Williams's thesis is ambiguous and we follow the Chapter 6 precedent.

---

## Number of evolutionary runs per condition

**Decision:** Target Williams's scale: 4 conditions × 10 runs × 500 generations × 50 individuals. Fallback if compute is constrained: 4 × 5 × 300 × 30.

**Reasoning:** The persistence infrastructure (atomic checkpointing, manifest, resumption) lets us run experiments overnight or across multiple machines without losing partial work, which makes the full Williams scale potentially achievable. If runtime measurements during the GA baseline check (Phase 6) suggest the full scale would exceed the time budget, we fall back to the scaled-down setting.

**Cost of scale-down (if we use it):** lower statistical resolution on the headline replication figure. Headline qualitative findings need to be robust to this; framed as a limitation in the discussion.

---

## Performance metric: Williams' ball-catching fitness (eq. 7.3)

**Decision:** Use Williams' fitness function from eq. 7.3, which combines (i) the reduction in horizontal displacement between agent and shape during the trial and (ii) the final horizontal distance at the moment of catch, averaged across trials in an evaluation and normalised.

**Reasoning:** Direct comparability with Williams' published curves. Any change to the fitness function obscures the replication. The methods section will describe the exact form (see `notes/methods_log.md` §7, to be written during the fitness pass).

**Note on previous documentation:** earlier drafts of this document and the methods log described only the second component (final distance); this was incomplete. The combined form is the correct one to implement.

---

## Genetic algorithm: tournament selection with elitism

**Decision:** Tournament selection (K = 3) with elitism of the single best individual. Asexual reproduction; Gaussian mutation only, no crossover. Real-valued genotype of length 35 in $[-1, 1]$, mutation with reflection at boundaries.

**Reasoning:** Williams uses fitness-proportional roulette wheel selection with elitism of the top 5. Chris's feedback on the proposal flagged this combination as "a direct route to premature convergence" — fitness-proportional selection concentrates reproduction on the early lead, and elitism guarantees that lead survives, collapsing diversity. Williams's task is soft enough that this didn't break his result, but using the same GA risks (a) reproducing a known anti-pattern in our submission, (b) confounding any genuine HP-induced search difficulty with GA-induced convergence problems.

Tournament selection is the EC textbook default for a reason: K controls selection pressure smoothly, the method is invariant to fitness scaling, and it does not concentrate reproduction on a single dominant individual. K = 3 is conventional and gives moderate selection pressure. Single-individual elitism prevents fitness regressions between generations without disabling the GA's ability to explore.

**Implications for results:**

- We are *not* exactly replicating Williams's GA. Our methods section states this and gives the EC-literature justification. The qualitative comparison with Williams becomes "we used a better-conditioned GA and still see qualitative effects X and Y", which strengthens rather than weakens the replication of the underlying biology.
- If our qualitative ordering differs from Williams's, this is itself an interesting result and should be reported. We do not need to match his fitness curves quantitatively to test the project's central claims (which are about HP dynamics, not GA dynamics).

**Trade-off explicitly considered:** running both GAs in parallel would let us isolate GA effects from HP effects. We do not have the time or report space for this. Flag as future work.

**GA scale:** unchanged by this decision. Population size is driven by genotype dimensionality and fitness noise, neither of which depends on the selection method.

---

## Baldwin effect: the framing for the introduction and discussion

**Decision:** Use the Baldwin effect as the primary theoretical frame for the project, with the Stolting et al. hypothesis as a complementary mechanistic story. Introduction cites Baldwin (1896), Hinton & Nowlan (1987), and Mayley (1996). Discussion reframes the frozen-HP test as a test of genetic assimilation.

**Reasoning:** The Baldwin effect — that lifetime adaptation can guide evolution, and that lifetime-acquired traits can become genetically assimilated over generations — is the canonical EC frame for situations like ours. Chris's feedback flagged that the Stolting paper "often crops up alongside the Baldwin effect" and was surprised the connection wasn't made. The connection is direct:

- **HP-during-development (condition 2)** is exactly the Hinton-&-Nowlan setup: lifetime learning before fitness evaluation. The Baldwin effect predicts this should help in early generations (because HP smooths the genotype-to-phenotype mapping) but the benefit should diminish as the population finds genotypes that don't need HP correction (genetic assimilation).

- **HP-during-behaviour (condition 3)** is the harder case where lifetime adaptation and fitness measurement are interleaved. The Baldwin effect prediction depends on whether genetic assimilation can happen: if it can, evolved genotypes become HP-independent and frozen-HP evaluations show small fitness drops; if it cannot, the population remains HP-dependent and frozen-HP evaluations show large drops.

- **The frozen-HP test, in this framing, is a direct empirical test of genetic assimilation.** Stolting's hypothesis (HP-during-behaviour evolves limit cycles in the joint state space) is a *specific mechanism* by which assimilation could fail — but the test itself is more general.

**Why this is the right framing:**

1. It places the project in the Sussex evolutionary-robotics tradition (Mayley's DPhil was at Sussex; Williams's thesis is in the same lineage).
2. It gives the introduction a cleaner motivating question than "Williams reports a result; we test Stolting's explanation."
3. It makes the discussion more substantive: we can interpret the frozen-HP drop size in terms of degree of assimilation, regardless of the specific mechanism.
4. Chris explicitly flagged this, so we know the marker will be reading for it.

**Methods are unchanged.** The Baldwin reframing is theoretical, not experimental. The same data answers both questions.

---

## Search dynamics, not "moving fitness landscape"

**Decision:** Drop the "moving fitness landscape" wording. Replace with "HP introduces an indirect genotype-phenotype mapping, which changes search dynamics across a fixed landscape."

**Reasoning:** The fitness landscape is the genotype → fitness mapping, which is deterministic for a given seed regardless of plasticity. What HP changes is the *phenotype that gets evaluated* — the genotype encodes initial weights and biases, and HP modifies them before fitness is measured. This is a feature of the encoding (indirect, via HP), not of the landscape (which is unchanged in shape).

The wording in the proposal was sloppy. Chris flagged it; the corrected framing is the one to carry forward into the report.

**Implication:** in analyses and discussion, talk about "how HP shapes the search trajectory" or "how HP changes the effective genotype-phenotype mapping", not "moving landscape" or "fitness landscape that changes during evaluation".

---

## What we are NOT doing

Spelled out so we don't drift back into them:

- **No discrimination task** in the main experiments
- **No duration sweep** of the developmental phase (Williams' Experiment 2 already covers this with 6000 timesteps)
- **No phototaxis or moving-light task** (Sandbox-driven idea that didn't survive the move to a custom simulator)
- **No $\rho$ functional-form variations** in the primary experiments
- **No Hebbian plasticity comparison** (interesting future work; out of scope)
- **No GA comparison** between Williams's GA and ours, despite this being a natural experiment — out of scope on time and report length

---

## Analyses Williams did not perform (the contribution)

Four analyses. Three concern what evolved (behavioural trajectories, per-neuron viable-range diagnostics, frozen-HP test); one concerns how it evolved (search-dynamics analysis). The search-dynamics analysis was added on 22 May after re-reading the assignment specification, which states that an evolutionary-algorithm project must analyse the larger adaptive system, including the EA, the problem space, the fitness function, and the population, and how the search is affected by them. The other three analyses are about the evolved controller and do not by themselves discharge that requirement; the search-dynamics analysis does.

### Behavioural trajectory analysis

For each condition, select representative individuals (best-of-best, median-of-best, perhaps worst-of-best). Plot agent x-position and shape (x, y)-position over time across a handful of representative trials. Overlay firing rates of all 5 neurons alongside. Compare strategies qualitatively across conditions.

**Why this matters:** Chris's group feedback explicitly called out "showing only learning curves and aggregate statistics without examples of actual learned/evolved behaviours is a route to a low mark." Williams' published figures are all fitness curves; no individual behaviours are shown in the conference paper, and only minimal examples in the thesis.

### Per-neuron viable-range diagnostics across evolution

For each condition, at each generation, take the best individual; run a representative trial; record for each neuron the fraction of timesteps its firing rate is in $[H_L, H_U]$; plot the resulting per-neuron-by-generation fraction over the course of evolution.

**Why this matters:** Williams' substrate-level claims (HP keeps neurons in the viable range) and his evolvability claims (HP-during-development helps) have not been directly connected to each other in the literature. This analysis bridges them: it asks whether the substrate-level effect persists through evolution, in each condition.

Connection to Baldwin: in conditions with HP, this metric tracks whether evolution comes to rely on HP-corrected firing rates (high in-range fraction maintained by HP) or evolves genotypes that produce in-range rates innately (high in-range fraction maintained without HP). The trajectory of this metric across generations is a direct view on the assimilation question.

### Frozen-HP test (the Stolting et al. test, reframed as an assimilation test)

For each evolved HP-during-behaviour individual (the `behaviour_only` and `both` conditions), let HP run so the controller settles into its behavioural dynamics, freeze the plasticity parameters at their current values, and continue the evaluation with no further plasticity. Measure the fitness drop relative to the same individual evaluated with HP active throughout.

**What "freeze" means.** Freezing sets $\dot w = \dot b = 0$ from the chosen instant onward, holding the weights and biases at whatever values HP has driven them to. It does *not* reset them to the genotype values. Williams calls this operation adiabatic elimination: a non-plastic CTRNN is instantiated from a plastic one by fixing the plastic variables at a point in time (thesis §6.2.3, "a non-plastic CTRNN can be instantiated from a plastic CTRNN by freezing the plastic variables at any particular point in time"). We use his term in the report.

**The freeze point (settled).** The test is faithful to Stolting et al. only if the system has first settled into its behavioural dynamics, then has the mechanism that sustains them removed. Stolting's own procedure is to let HP act for a settling period (500 seconds in their CPG study), then freeze and test for persistence of the behaviour (Stolting et al. 2023, p. 4). We follow this:

- **`both`:** run the full 6000-timestep developmental phase, then run into the trial, then freeze. The natural settling point is the end of the developmental phase; freezing there is the primary variant.
- **`behaviour_only`:** there is no developmental phase, so HP runs from the genotype during the trial. Allow a settling window into the trial, then freeze for the remainder. State the window length and confirm the result is not knife-edge sensitive to it.

**Controls (noise floor).** Re-evaluate each `dev_only` and `no_hp` individual under the identical freeze-and-continue procedure. Both ran with HP off during their behaviour already, so the freeze should produce approximately zero fitness change. A non-trivial drop in either control is a bug signal, not a result, and is checked before any `behaviour_only`/`both` drop is interpreted.

**Within-individual variance baseline.** The drop for a single individual is only meaningful against the noise of fitness evaluation itself. Re-evaluate each individual N times with different shape-sequence seeds, HP active, to estimate within-individual fitness variance. The freezing effect is significant if it exceeds this baseline.

**Interpretation.**

- *Stolting framing:* a large drop is consistent with the behaviour depending on ongoing HP dynamics (Stolting's HP-enabled limit cycles, here in the joint network-and-parameter state space); a small drop is inconsistent with that mechanism.
- *Baldwin framing:* a large drop indicates the population has *not* genetically assimilated the adaptive behaviour, it remains HP-dependent; a small drop indicates assimilation has occurred. Either result is a substantive finding.

This is the project's clearest single scientific question.

**Why this is a single test, not a sweep.** An earlier plan considered a second variant that freezes from the genotype (HP never allowed to act), reading "is the bare genotype competent?" as a separate assimilation measure. We do not run this as a primary analysis. The genotype reading is largely already available from the evolutionary conditions: a `behaviour_only` genotype evaluated with HP entirely off is close to evaluating that genotype as a `no_hp` agent, so comparing final fitness across the four evolved conditions already carries most of the innate-competence signal. Adding a separate genotype-freeze sweep would duplicate that information and dilute the report's focus, which the assignment specification asks us to keep on one or two main points. The genotype/assimilation reading is therefore handled in discussion by relating the frozen-HP drop to the cross-condition comparison, not by a parallel freeze analysis. The genotype-freeze variant survives only as an optional extension (Phase 9) if we find we have too little to discuss, which a 3000-word report plus methods makes unlikely.

**Novelty check (verified against the sources).** Williams's Chapter 7 has four experiments: comparative HP schemes during behaviour, the developmental-period experiment (which is our `dev_only` condition, evolved under HP-then-frozen), a comparison with centre-crossing networks, and larger non-plastic networks. None of them takes online-HP-evolved individuals and freezes HP post-hoc to measure a drop. The frozen-HP test on Williams's ball-catching agent is therefore genuinely new, as the proposal claims.

### Search-dynamics analysis (analysis of the larger adaptive system)

The other three analyses examine the evolved controller. This one examines the search that produced it: how the population moves through the space across generations, and how HP changes that movement. All of it is computed from data already saved (`history/gen_NNNN.npz` holds the full population fitness array per generation, not only the best), so no re-running is required.

The material splits cleanly along the replicate/extend line, and the report places the two halves in different sections accordingly.

- **Replication half (belongs with the Phase 7 replication results).** Williams reports that plastic networks show quicker progress early in the evolutionary run and greater consistency in reaching a reasonable level of performance, even though they are eventually out-performed (thesis Chapter 7 results and §8.2.5). These are claims about the search, and reproducing them is part of replicating Williams. The best-fitness-per-generation curves with their across-run spread, read for early-generation progress and run-to-run consistency, test whether this pattern holds under our better-conditioned GA.

- **Extension half (a distinct Phase 8 analysis).** Williams reports best-fitness only, not population-level behaviour. Plotting the population fitness distribution and a diversity measure over generations, per condition, is something he did not do. This is where the better-conditioned GA claim becomes visible (tournament versus roulette-plus-top-5) and where the "HP changes search dynamics across a fixed landscape" framing gets its evidence rather than remaining an assertion.

**Why this matters:** the assignment specification is explicit that an EA project's analyses should focus on the larger adaptive system, naming the EA, the problem space, the fitness function, the population, and the fitness landscapes they codetermine. Without this analysis the search-dynamics discussion point has no result behind it; with it, the report covers both axes an EA project is expected to cover, what evolved and how.

**Implementation note:** the replication-versus-extension split is a presentation boundary, not a data boundary. Both halves read the same `history/` arrays, so in code this is most likely one data-loading layer feeding two figures, and probably the same Claude Code pass that produces the Phase 7 replication figure. The split governs where results live in the report, not how many things get built.

---

## Compute budget and parallelisation

Each fitness evaluation: 5 neurons × ~1500 timesteps × ~10 trials per evaluation × 20 shapes per trial. Approximately 0.1–0.5 seconds per evaluation depending on implementation efficiency.

Full Williams-scale experiment: 4 conditions × 10 runs × 500 generations × 50 individuals ≈ $10^6$ evaluations. Single-threaded at 0.3s each is ~80 hours. Tractable across multiple overnight runs with multi-core parallelisation, or on a single machine with `multiprocessing` (Williams' fitness function is trivially parallelisable across population members).

Scaled-down fallback: 4 × 5 × 300 × 30 = 180,000 evaluations ≈ 15 hours single-threaded.

**Plan:** measure single-threaded timing during the GA baseline check. Decide between Williams-scale and scaled-down based on what the timing shows. Parallelise across cores in either case if it's a clean win.

---

## Results presentation: SD on curves, points on the final comparison

**Decision:** On the per-generation fitness curves, show the mean across the 5 runs with a shaded band of $\pm 1$ standard deviation, and overlay the individual run curves faintly. On the final-fitness comparison, show a box plot with all 5 run values overlaid as points, and no error band.

**Reasoning:** Standard deviation describes the run-to-run spread of the data and does not shrink with sample size; standard error of the mean (SD/$\sqrt n$) describes the precision of the mean estimate and is roughly half the width of the SD band at $n = 5$. An SEM band on five highly variable runs would visually imply a separation between conditions that the raw points do not support. SD is the honest summary of variability here. For the final comparison, with only five runs per condition, the most honest display is the five points themselves; a summary band is precision the sample does not earn. State "mean $\pm$ 1 SD across 5 runs" in the curve caption.

**Note:** a $\pm$ symbol in a figure caption broke section parsing in the Assignment 1 word count (interpreted as a display-math delimiter). Watch for this when finalising captions.

---

## Reproducibility

Each evolutionary run is initialised from a different seed, recorded in the run's manifest entry. Raw per-generation data (best genotype, fitness statistics) saved as compressed numpy arrays. All plots regenerated from saved data, not from live runs.

This was a hard lesson from Assignment 1: keep the analysis and plotting code separate from the experiment runner so a failed plot doesn't require a re-run of the experiment.
