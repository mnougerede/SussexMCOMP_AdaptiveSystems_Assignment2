# Assignment 2 — Submission checklist
**Deadline: 28 May 2026, 4 pm.**

---

## STATUS: What is done

### Empirical and analysis — complete ✓
- All 20 runs at n=10. All analysis scripts run. All figures exist.

### Report — all sections drafted and updated ✓
- All five sections (intro, methods, results, discussion, abstract) drafted and revised
- Word count: 2575 / 3000 (abstract 134, intro 615, results 883, discussion 943)
- Methods unbudgeted: 1647
- All known factual errors corrected in this session
- Author → Candidate 262983, date removed

### Notes — corrected ✓
- `methods_log.md` and `design_decisions.md` updated

---

## REMAINING TASKS — in order

### 1. FIXES NEEDED NOW (blocking or factual)

#### 1a. Search dynamics prose — WRONG, must fix
The current results §3.7 says "the population fitness spread collapses to near
zero by roughly generations 10 to 20 in every condition." **This is false.**
The CSV shows:
- Behaviour only: spread drops below 0.02 by generation 3
- Both: spread drops below 0.02 by generation 4
- No HP and Dev only: spread NEVER drops below 0.02 — final spread ~0.11 and ~0.12

The "all four converge similarly" framing is wrong. What's actually true and
interesting: the two online-HP conditions (Behaviour only, Both) converge much
more rapidly and tightly than No HP and Dev only. Dev only starts high (gen 0
best = 0.619 vs ~0.51 for others — because development runs before the first
evaluation) and remains diverse. No HP also stays diverse. This is a real
finding worth reporting accurately.

The `[CHECK]` flag in the updated results.tex points to this; need to rewrite
the search dynamics paragraph with accurate numbers before compiling.

#### 1b. Figure 6 caption — SD units vs raw fitness
The caption says "drop magnitude per individual in units of the
within-individual measurement-noise SD" but the figure y-axis says "Fitness
drop (HP-active minus HP-frozen)" — raw fitness units. Fix the caption.

#### 1c. Fitness function equation — needs Claude Code check
The per-shape score equation added to methods §2.6 has a `[CHECK]` comment
because the exact formula wasn't verified against the code. Must confirm before
submitting. See the Claude Code prompt list below.

#### 1d. Abstract — update to remove SD units language
The abstract says "drops fitness far beyond the measurement-noise baseline" —
this is fine. But double-check no specific numbers in the abstract reference
SD units that are no longer in the results.

---

### 2. BIB FILE FIXES

The following are needed in `references.bib`:

- [ ] `dipaolo2000`: change `@article` to `@inproceedings`, add:
  `booktitle = {From Animals to Animats 6: Proceedings of the Sixth International Conference on Simulation of Adaptive Behavior},`
  `publisher = {MIT Press},`
  `pages = {417--426},`
  (These details are already in the compiled bibliography — they came from the
  PDF so the bib entry must already have them. Check the current bib file.)

- [ ] `williams2005`: currently has `[VERIFY]` comment — no pages or exact
  proceedings string. Either add the details or decide to leave it uncited
  (it is currently uncited so biblatex will omit it silently).

- [ ] Three uncited entries: `williams2005`, `beer1996`, `turney1996`.
  `beer1996` could reasonably be cited in methods §2.5 where the ball-catching
  task is introduced. `williams2005` and `turney1996` can be deleted if not
  cited. Decide and act.

- [ ] Remove all `[VERIFY]` comment lines from the bib — these are scaffolding
  for the drafting process and should not be in the submitted file.

---

### 3. CODEBASE TIDY — remove AI scaffolding

The submission codebase should be human-oriented. Things to remove or revise:

- [ ] Comments in source files that are clearly written for/by an AI coding
  agent (e.g. "Claude Code prompt", "NOTE FOR CLAUDE", explicit instructions
  about what a coding agent should do next, etc.)
- [ ] Any TODO/FIXME comments that describe scaffolding rather than genuine
  known issues (a real known-issue comment is fine)
- [ ] The `[CHECK]`, `[PSEUDOCODE]`, `[VERIFY]` markers in the LaTeX files —
  these are drafting scaffolding. Either resolve them or delete them before
  submitting the report source
- [ ] `batches/` versioned provenance records: check these are clean and
  don't contain agent-conversation artifacts
- [ ] `notes/` directory: methods_log and design_decisions are legitimate;
  remove any files that are clearly AI working notes rather than project
  documentation
- [ ] README.md: read through and make sure it reads as a human-written
  project README, not as agent instructions

Standard academic codebase items:
- [ ] `git status` — commit outstanding changes
- [ ] `uv run pytest` — all tests pass
- [ ] Confirm `main` is submission branch
- [ ] `git tag submission && git push origin submission`
- [ ] Confirm repo is public or accessible to Chris

---

### 4. COMPILE AND READ

- [ ] Delete `.aux`, `.bbl`, `.bcf`, `.blg` before compiling (clears Stolting
  umlaut cache and any other stale artefacts)
- [ ] `pdflatex → biber → pdflatex → pdflatex`
- [ ] Check no `[CHECK]` or `[VERIFY]` comments appear in the PDF (they
  shouldn't — they're LaTeX comments — but confirm)
- [ ] Check for overfull hbox warnings
- [ ] **Read the compiled PDF end to end.** This is the most important step.

---

## CLAUDE CODE PROMPT LIST

Items that need code inspection before the report can be finalised:

**Priority 1 — blocking:**

1. Fitness function formula: What is the exact per-shape score formula in
   `src/environment/fitness.py`? Specifically: how are D0 and Df combined into
   the per-shape score, and is there a normalisation by Smax? Show the relevant
   code lines and express the formula mathematically.

2. Search dynamics — spread behaviour confirmation: The CSV shows Behaviour
   only and Both converge (spread < 0.02) by generation 3–4, while No HP and
   Dev only never converge (final spread ~0.11–0.12). Can you confirm this is
   real and not a data artefact? Specifically: are the spread values in
   search_dynamics_summary.csv the mean across 10 runs, or the spread from a
   single run? (The CSV column is `mean_spread` — confirm this is the mean of
   the per-run spreads, not the spread of the per-run final fitnesses.)

**Priority 2 — codebase tidy:**

3. AI scaffolding audit: Search the codebase for comments or strings that look
   like they were written for/by an AI coding agent — things like "NOTE FOR
   CLAUDE", "Claude Code prompt", explicit step-by-step instructions inside
   comments that read more like prompts than documentation. List any files and
   line numbers where these appear.

4. Check `notes/` directory contents: List all files in the `notes/` directory.
   Which ones look like genuine project documentation vs AI working notes that
   shouldn't be in a submission?

---

## KEY NUMBERS (verified against actual outputs)

- Substrate: 86.1% → 53.2% outside [HL, HU] during HP; 59.4% after HP off
- Final fitnesses (n=10): No HP 0.674 (SD 0.096), Dev only 0.838 (SD 0.059),
  Behaviour only 0.709 (SD 0.097), Both 0.675 (SD 0.077)
- KW H=17.71, p=0.0005. Dev only sig. > all others (Bonferroni).
- Frozen-HP: raw drops confirmed in figures. 5/10 Dev only collapses, 3/10
  Behaviour only, 3/10 Both.
- Viable-range gen 199 HP on: frac_V 0.43–0.45 for HP conditions, 0.19 No HP.
  HP off: Dev only and Behaviour only ~0.08, Both 0.16.
- Search dynamics (from CSV): Dev only gen 0 best = 0.619 (development effect
  visible immediately); Behaviour only and Both spread collapses by gen 3–4;
  No HP and Dev only maintain spread ~0.10–0.12 throughout.

### GA spec
Tournament K=3, elitism 5 (matching Williams), no crossover, Gaussian mutation
N(0, 0.01), p_m=0.03, boundary reflection at [-1,1].
