# Assignment 2 — CTRNN Homeostatic Plasticity: To-Do List

**Submission deadline:** 28 May 2026 (late period), 4 pm.
**Today:** 27 May 2026. ~24 hours remain.

---

## What is done

- **Phases 0–8:** All empirical and analysis work is complete at n=10.
- **Phase 10 infrastructure:** `report.tex` (master), `sections/methods.tex` (draft), `references.bib` (14 entries). Multi-section `\input` structure in place.
- **Methods section:** drafted and static-validated (equations 1–9 balanced, all cite keys present, cross-refs clean). Handed to Max for review.

---

## What still needs doing — ordered by criticality

### TODAY (report writing — the critical path)

These are the things that must happen before 4 pm tomorrow. Writing order: results → analyses prose → discussion → introduction → abstract. Methods is already drafted.

#### 1. Review and finalise `sections/methods.tex`

- [ ] Read the draft and correct anything wrong. Key items to check:
  - [ ] `[CHECK]` shape radius and initial height — confirm against your `config.json` or `shapes.py`
  - [ ] `[CHECK]` fitness function: confirm eq. 7.3 form against `src/environment/fitness.py` — the draft describes it correctly per your design decisions but you should verify the $S_{\max}$ formulation
  - [ ] `[CHECK]` 3 trials per evaluation (scale decision §7d) — draft says "several"; fill in the exact number
  - [ ] System diagram placeholder — figure commented out; insert once you have the diagram, or cut the placeholder if you won't draw it in time
  - [ ] GA pseudocode — `[PSEUDOCODE]` comment in the draft; add an `algorithm2e` block or remove the comment
- [ ] Once happy, this section is done

#### 2. Write `sections/results.tex`

Create `sections/results.tex`, `\input` it in `report.tex`. All figures already exist. This is mostly: one figure, one paragraph each. Suggested subsection order matching the report scaffold:

- [ ] **Substrate-level sanity check** — `figs/substrate_check.pdf`; one short paragraph: 86.1% → 47.7% → 53.9%. Foreshadows frozen-HP test.
- [ ] **Replication: four-condition evolvability** — `figs/replication_fitness_curves.pdf` + `figs/replication_final_box.pdf`; KW H=17.71, p=0.0005; Dev only sig. > all at Bonferroni. Note the online-HP-not-worse divergence from Williams.
- [ ] **Search replication** — dev\_only quicker early progress visible in search\_dynamics figure; record this as part of the replication read.
- [ ] **Behavioural trajectories** — 2–4 panels from `figs/behavioural_trajectories.pdf`; blue/cream/red meaning in caption.
- [ ] **Viable-range diagnostics** — compact figure (if built); key numbers from methods\_log §11.5.
- [ ] **Frozen-HP test** — `figs/frozen_hp_scatter.pdf` + `figs/frozen_hp_drop.pdf`; complete collapse individuals noted; assimilation verdict stated here.
- [ ] **Search dynamics** — `figs/search_dynamics_population.pdf`; conditions differ in *where* they converge, not whether/how fast.

#### 3. Write `sections/discussion.tex`

Create `sections/discussion.tex`, `\input` it in `report.tex`. Pre-planned in methods\_log §11.4 and §11.5 — the discussion almost writes itself from those sections. Planned thread order:

- [ ] Assimilation verdict as the headline (MP1): HP is constitutive, not merely helpful — Baldwin frame payoff
- [ ] Dev only paradox (MP2): best-performing yet most HP-dependent; selection regime explanation
- [ ] Stolting mechanism as candidate explanation for assimilation failure
- [ ] Search-dynamics framing: HP changes the genotype-phenotype map, not the fitness landscape
- [ ] Adaptivity revisited: which timescales are active per condition; Ashby/ultrastability mapping
- [ ] Divergence from Williams: online-HP not worse — GA-conditioning candidate explanation
- [ ] Limitations (one honest paragraph): single task, fixed timescales, n=10, no discrimination, developmental-settling confound
- [ ] Future work: discrimination; timescale-separation sweep; alternative ρ shapes; exact-Williams-GA comparison

#### 4. Write `sections/introduction.tex`

Create `sections/introduction.tex`, `\input` it in `report.tex`. Write this after discussion so you know exactly what needs to be planted here.

- [ ] Adaptivity definition — Di Paolo grounded in Ashby
- [ ] HP background — biological (Turrigiano) + computational substrate-level gains (Williams & Noble 2007)
- [ ] Substrate-vs-evolvability tension as the motivating question
- [ ] Baldwin-effect framing: lifetime adaptation can guide evolution; assimilation as the question
- [ ] Contribution statement: replication + four analyses, central one being the frozen-HP / assimilation test

#### 5. Write the abstract

- [ ] One paragraph, ~150 words. Write last. Lead with the assimilation verdict.

---

### BEFORE SUBMITTING (polish and codebase)

These can be done in parallel with or just after writing. They are not blocking the writing, but they are required for submission.

#### Figure polish (one remaining item)

- [ ] **Viable-range compact figure** — `scripts/analysis/viable_range_compact.py`. Check whether this already ran and exists at `figs/viable_range_compact.pdf`. If not, run it. This is the report-body figure for §8b.
- [ ] `[VERIFY]` The `EXPECTED_RUNS` fix in `replication_figure.py` — todo says "should be updated to expected 10"; check whether this was done in the figure-polish pass that completed Phase 8.

#### Final polish (report)

- [ ] All figures referenced from prose by `\ref`
- [ ] All equations referenced by number in the text
- [ ] All citations in `references.bib`; no uncited entries; no unreferenced citations
- [ ] `[VERIFY]` four bib entries in `references.bib` have `[VERIFY]` markers; confirm against sources:
  - `williams2005` — exact proceedings title/pages
  - `williamsnoble2007` — title matches the scanned PDF
  - `stolting2023` — author first names and proceedings details
  - `dipaolo2000` — venue is SAB 2000 proceedings (`@inproceedings`), confirm pages
- [ ] LaTeX quote characters: `''text''` not `"text"`
- [ ] Compile clean: `pdflatex → biber → pdflatex → pdflatex`, no errors or overfull hboxes that break layout

#### Codebase tidy

- [ ] `git status` — no uncommitted working-tree changes before tagging
- [ ] Final `pytest` run — all tests passing on `main`
- [ ] Check `main` branch is the submission branch
- [ ] Tag the submission commit: `git tag submission`
- [ ] Push tag: `git push origin submission`
- [ ] Verify the GitHub repo is public (or accessible to Chris)

#### System diagram

- [ ] Draw the system diagram (three coupled loops: sensorimotor, HP parameter-update, GA selection). Either:
  - Insert as `figs/system_diagram.pdf` and uncomment the `\includegraphics` in `sections/methods.tex`, or
  - Accept the figure is missing and remove the placeholder comment from the methods draft

---

## Items NOT on the critical path (do not spend time on these)

- Phase 9 optional extensions (oscillation analysis, genotype-freeze variant)
- Discrimination task
- Timescale-separation sweep
- Exact-Williams-GA comparison
- Further runs

---

## Reference snippets (unchanged — for writing reference)

### Key numerical results

- **Substrate check:** 86.1% → 47.7% outside [H_L, H_U] during HP; 53.9% after HP off
- **Final fitnesses (n=10):** No HP 0.674 (SD 0.096), Dev only 0.838 (SD 0.059), Behaviour only 0.709 (SD 0.097), Both 0.675 (SD 0.077)
- **KW:** H=17.71, p=0.0005. Dev only sig. > all others (Bonferroni p_bonf=0.0035 vs No HP and Both, p_bonf=0.0217 vs Behaviour only).
- **Frozen-HP drops:** No HP ±0.85 SD (noise); Dev only 5.6–21.2 SD (2 complete collapses); Behaviour only 1.0–5.3 SD (all positive); Both bimodal, 2 complete collapses + 3 moderate drops.
- **Viable-range (gen 199, HP on):** all HP conditions frac_V 0.34–0.44 vs No HP 0.15. HP off: collapse to 0.04–0.08.

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

Tournament selection $K = 3$, elitism of 1, no crossover, Gaussian mutation per allele $\mathcal{N}(0, \sigma_m^2)$, $\sigma_m = 0.1$, $p_m = 0.03$, reflection at $[-1, 1]$.
