# Assignment 2 — Submission checklist
**Deadline: 28 May 2026, 4 pm. ~5 hours remain.**

---

## STATUS: What is actually done

### Empirical work — complete
- All 20 runs (4 conditions × 5 runs × 2 batches = 10 per condition) at n=10. ✓
- All analysis scripts run: replication figure, search dynamics, behavioural trajectories, viable-range diagnostics, frozen-HP test. ✓

### Report — all sections drafted
- `report.tex` — master with `\input` wiring for all four sections ✓
- `references.bib` — 14 entries ✓
- `sections/methods.tex` — drafted, static-validated, corrected (elitism 5, n_trials 10, without replacement) ✓
- `sections/results.tex` — drafted, static-validated ✓
- `sections/discussion.tex` — drafted, static-validated ✓
- `sections/introduction.tex` — drafted, static-validated ✓
- Abstract — drafted inline in `report.tex` ✓
- Word count: 2708 / 3000 (~290 words headroom) ✓

### Notes — corrected
- `methods_log.md` — elitism, n_trials, pop_size, replacement, run count, Williams initials all corrected ✓
- `design_decisions.md` — GA elitism and reasoning corrected ✓

---

## REMAINING TASKS — in order of priority

### 1. FIGURES — run missing scripts first (do this now)
These are blocking because the report references them by filename.

| Figure file | Status | Action |
|---|---|---|
| `figs/substrate_check.pdf` | **MISSING** | Run the substrate check script (check exact name in `scripts/`) |
| `figs/viable_range_compact.pdf` | **UNKNOWN** | Check if it exists. If not: `uv run python scripts/analysis/viable_range_compact.py` |
| `figs/replication_fitness_curves.pdf` | Should exist | Confirm |
| `figs/replication_final_box.pdf` | Should exist | Confirm |
| `figs/behavioural_trajectories.pdf` | Should exist | Confirm |
| `figs/frozen_hp_scatter.pdf` | Should exist | Confirm |
| `figs/frozen_hp_drop.pdf` | Should exist | Confirm |
| `figs/search_dynamics_population.pdf` | Should exist | Confirm |

Also check: does `replication_figure.py` have `EXPECTED_RUNS = 10`? If it still says 5, fix it and regenerate.

Once all figures are confirmed on disk, uncomment the `\includegraphics` lines in `sections/results.tex` (6 figure environments, all currently commented out).

---

### 2. METHODS — two remaining checks (quick)

- [ ] Confirm shape radius and initial height in `sections/methods.tex` match your config. The draft says radius 10, spawn y=100 (from methods log §6) — verify these are correct.
- [ ] System diagram decision:
  - **If yes (time permitting):** save as `figs/system_diagram.pdf`, uncomment the `\includegraphics` block in `sections/methods.tex` §2.1.
  - **If no (recommended):** delete the entire commented-out `\begin{figure}...\end{figure}` block in `sections/methods.tex` §2.1. A missing diagram is a minor deduction; a broken comment left in is avoidable noise.
- [ ] GA pseudocode block: currently a comment only — will not appear in PDF. No action required. Delete the comment if you want to tidy the file.

---

### 3. CONTENT REVIEW PASS — section by section

Read each file and check the specific items below.

#### `sections/introduction.tex`

Logical flow:
- [ ] Para 1: adaptivity definition clearly states "maintaining essential variables within viable bounds across multiple timescales." This exact phrase is referred back to in the discussion.
- [ ] Para 2: HP established as cybernetic negative feedback before the evolvability question is raised. The discussion uses that exact phrase — it must be planted here.
- [ ] Para 3: the claim "HP left running during behaviour did not improve evolved fitness, and could make matters worse" — verify this is what Williams reports, not what your replication found. Your data shows Both ≈ No HP (not worse). Make sure you are describing Williams's result here, not your own.
- [ ] Para 5 final sentence: "the result --- that assimilation has not occurred in any HP condition --- organises the discussion that follows." Decide whether to keep (previews verdict, standard science writing) or cut (withholds result for suspense). Either is defensible.

Language:
- [ ] `\emph{developmental}` and `\emph{during}` (para 3, contrast pair) — consider rewriting to let sentence structure carry the contrast rather than italics: "HP applied as a developmental phase improved evolved performance; HP running during behaviour did not."
- [ ] Em-dashes: 5 uses, all legitimate parenthetical. ✓

#### `sections/results.tex`

Logical flow:
- [ ] Substrate check: foreshadows frozen-HP test at end of paragraph. ✓
- [ ] Frozen-HP paragraph: collapse counts (5/10 Dev only, 3/10 Behaviour only, 3/10 Both) — cross-check these against `figs/frozen_hp_results.csv` before submitting. These are specific numbers in the report.
- [ ] Frozen-HP paragraph: the todo reference snippets say "Dev only 5.6–21.2 SD (2 complete collapses)" but the methods log §11.5 says "5/10 collapses." The draft uses 5/10. Confirm which is correct.

Language:
- [ ] `\emph{do}` in "what the evolved controllers actually do" — casual register; consider cutting the italics.
- [ ] `\emph{when the plasticity acts before evaluation}` (line 103) — over-long for emphasis. Rewrite as plain prose.
- [ ] Condition names (`\emph{Dev only}` etc.) used consistently throughout. ✓

#### `sections/discussion.tex`

Logical flow:
- [ ] Opens by paying off the Baldwin frame from the intro. ✓
- [ ] Refers back to the adaptivity definition by name. ✓
- [ ] Settling confound acknowledged in one paragraph. ✓
- [ ] "HP does not change the shape of the fitness landscape, which is fixed by the evaluation procedure" — this is the key correction to the proposal's "moving landscape" wording. ✓
- [ ] Future-work paragraph: four directions. Check you are comfortable defending all of them if asked.
- [ ] The Both puzzle explanation: "ongoing plasticity works against the developmentally established configuration" — stated as most parsimonious, not demonstrated. ✓

Language:
- [ ] Em-dash used as colon substitute in the adaptivity paragraph: "timescales --- the HP conditions are more adaptive." If you want a more formal register, replace with a colon.
- [ ] "in a precise and somewhat uncomfortable sense" — good intellectual honesty signal. Keep.
- [ ] "load-bearing rather than corrective" — good concrete image. Keep.

#### `sections/methods.tex`

- [ ] Shape geometry: radius 10, spawn y=100 — confirm correct.
- [ ] §8.7 justification: now says we match Williams on elitism and mutation rate, differ only on selection. ✓
- [ ] The `[CHECK]` and `[PSEUDOCODE]` markers are comments — they will not appear in the compiled PDF. You can leave or delete them.

---

### 4. FORMAT PASS — mechanical

- [ ] **Condition name consistency.** Every occurrence of No HP / Dev only / Behaviour only / Both as a condition label should be `\emph{}`. Check introduction and discussion match the consistent treatment in results.
- [ ] **Hyphenation consistency.** "Behaviour only" (label, no hyphen) vs "behaviour-only HP" (adjective, hyphen). Pick one pattern and apply it throughout.
- [ ] **Figure captions.** Before uncommenting `\includegraphics` lines, check each caption is self-contained: states what SD bands mean, states colour scheme for heatmaps, wraps any `\pm` in `$...$`.
- [ ] **No straight double-quotes in LaTeX source.** Run: `grep -n '"' sections/*.tex` — any outside math/comments should be ` ``text'' `.
- [ ] **`dipaolo2000` bib entry type.** Currently `@article` — should be `@inproceedings`. Fix: change to `@inproceedings` and add `booktitle = {From Animals to Animats 6: Proceedings of the Sixth International Conference on Simulation of Adaptive Behavior}` and `publisher = {MIT Press}`.
- [ ] **Uncited bib entries:** `williams2005`, `beer1996`, `turney1996` are in the bib but not cited. `biblatex` will silently omit them — no error. Consider adding `\parencite{beer1996}` in the environment/task description in methods, where the ball-catching task is introduced.
- [ ] **Stolting first name:** the bib has "Jordan Stolting" — verify this is correct.
- [ ] **`\parencite` vs inline citation style.** Where "Williams" is the subject of a sentence, the citation should appear at the end of the clause in parentheses, not interrupt the prose. Spot-check the introduction.

---

### 5. COMPILE

- [ ] Delete any stale aux files (`.aux`, `.bbl`, `.bcf`, `.blg`) for a clean build.
- [ ] `pdflatex → biber → pdflatex → pdflatex`
- [ ] Check for overfull hboxes that break layout (warnings in the log).
- [ ] Read the compiled PDF end to end — this is the most important review step. You will catch things in the PDF that you miss in the source.

---

### 6. CODEBASE TIDY (do last, ~15 minutes)

- [ ] `git status` — commit any outstanding changes
- [ ] `uv run pytest` — all tests passing
- [ ] Confirm `main` is the submission branch
- [ ] `git tag submission && git push origin submission`
- [ ] Verify repo is public (or accessible to Chris)

---

## FIGURES REFERENCE — complete list

Every figure the report references, with label and expected filename:

| Label in report | File(s) needed | Section |
|---|---|---|
| `fig:substrate` | `figs/substrate_check.pdf` | Results 3.1 — **MISSING, must generate** |
| `fig:replication` | `figs/replication_fitness_curves.pdf` + `figs/replication_final_box.pdf` | Results 3.2 |
| `fig:trajectories` | `figs/behavioural_trajectories.pdf` | Results 3.4 |
| `fig:viablerange` | `figs/viable_range_compact.pdf` | Results 3.5 — **CHECK exists** |
| `fig:frozenhp` | `figs/frozen_hp_scatter.pdf` + `figs/frozen_hp_drop.pdf` | Results 3.6 |
| `fig:searchdynamics` | `figs/search_dynamics_population.pdf` | Results 3.7 |

System diagram (`figs/system_diagram.pdf`) referenced in methods — currently commented out. Decision required (see task 2).

All six result figures are inside `\begin{figure}...\end{figure}` blocks in `sections/results.tex` with `\includegraphics` lines commented out. After confirming files exist, uncomment each `\includegraphics` line.

---

## TIME ESTIMATE

| Task | Est. time |
|---|---|
| Generate missing figures, confirm all exist | 20 min |
| Methods geometry check + diagram decision | 15 min |
| Content review pass (all sections) | 60 min |
| Format pass | 30 min |
| Compile clean, fix errors | 20 min |
| Read compiled PDF end to end | 30 min |
| Codebase tidy + tag + push | 15 min |
| **Total** | **~3 hours** |

~2 hours buffer. Use it if tasks run over. Do not spend it adding new content.

---

## STALE REFERENCE — fix in this file
The reference snippets below still say "elitism of 1" in the GA spec. This is wrong — the actual runs used elitism of 5. Update this line when you update this file.

### Key numerical results
- Substrate check: 86.1% → 47.7% outside [H_L, H_U] during HP; 53.9% after HP off
- Final fitnesses (n=10): No HP 0.674 (SD 0.096), Dev only 0.838 (SD 0.059), Behaviour only 0.709 (SD 0.097), Both 0.675 (SD 0.077)
- KW: H=17.71, p=0.0005. Dev only sig. > all others (Bonferroni p_bonf=0.0035 vs No HP and Both, p_bonf=0.0217 vs Behaviour only).
- Frozen-HP drops: No HP within ±1 SD (noise); Dev only 5.6–28.8 SD, 5/10 complete collapses; Behaviour only 0.2–17.4 SD, 3/10 complete or near-complete collapses; Both 3/10 complete collapses, 4/10 moderate drops.
- Viable-range (gen 199, HP on): all HP conditions frac_V 0.43–0.45 vs No HP 0.19. HP off: Dev only and Behaviour only collapse to ~0.08; Both retains 0.16.

### GA spec (corrected)
Tournament selection K=3, elitism of 5 (matching Williams), no crossover, Gaussian mutation per allele N(0, sigma_m^2), sigma_m=0.1, p_m=0.03, reflection at [-1,1].
