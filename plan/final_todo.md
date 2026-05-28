# Assignment 2 — Final submission checklist
**Deadline: 28 May 2026, 4 pm. ~80 minutes remaining.**

---

## WHAT IS DONE

### Report — complete ✓
All sections drafted, revised, and verified against actual data files.
- Word count: ~2575 / 3000
- All figures present and wired in
- All numerical claims verified against CSVs (replication stats, frozen HP, viable range, search dynamics)
- Fitness function equations corrected and verified against code
- Bibliography clean: 11 entries, all cited, no scaffolding comments, dipaolo fixed to @inproceedings
- Author = Candidate 262983, no date
- Stolting: no umlaut ✓
- "The one methodological difference" → "the main methodological difference" ✓
- Figure 7 caption fixed (dotted line, not shaded band) ✓
- Gen-0 comparison corrected (0.508–0.525 range) ✓

---

## WHAT REMAINS — in order of priority

### 1. COMPILE THE REPORT (do now — ~10 minutes)

```
cd report/
rm -f *.aux *.bbl *.bcf *.blg *.log *.run.xml
pdflatex report.tex
biber report
pdflatex report.tex
pdflatex report.tex
```

Check the output for:
- [ ] No LaTeX errors (warnings are usually fine)
- [ ] All figures appear — confirm pages 7–16 have Figures 1–7
- [ ] No figure placeholders or "[Placeholder]" text
- [ ] References section renders correctly (11 entries, no "Stölting")
- [ ] Page count reasonable (~16 pages)

### 2. READ THE COMPILED PDF (15 minutes)

This is the most important remaining task. Read it as a marker would.

Specific things to check that I cannot verify from source:
- [ ] Figure 5 right panel (entry-exit bar chart): confirm the HP-off bars ARE
  visible for the three HP conditions. Earlier concern was they might be
  invisible. If they're not visible, the caption needs adjusting.
- [ ] Figure 6 top panel: confirm the No HP points are clustered near the y=x
  line (small or zero drops), and HP conditions show clear below-diagonal drops.
- [ ] Figure 7: confirm the dotted spread lines are visible and the Behaviour
  only and Both panels do show near-zero spread, while No HP and Dev only show
  substantial spread throughout. The caption now says "dotted line" — make sure
  it matches what the figure shows.
- [ ] The system diagram (Figure 1) renders cleanly and the caption has no
  placeholder text.
- [ ] Section numbering is correct throughout.
- [ ] No widows/orphans or broken equations that affect readability.

### 3. CODEBASE TIDY (~30 minutes)

The codebase needs to read as a human-produced research project, not as
AI working notes. In order of importance:

- [ ] Ask Claude Code to run the scaffolding audit (prompt 3 from the todo):
  "Search the codebase for comments that look like they were written for or by
  an AI coding agent — phrases like NOTE FOR CLAUDE, Claude Code prompt, or
  step-by-step instructions inside comments that read more like prompts than
  documentation. List files and line numbers."

- [ ] Check `notes/` directory. The `methods_log.md` and `design_decisions.md`
  are legitimate project documentation and should stay. Remove any files that
  are clearly AI working notes.

- [ ] Remove `[CHECK]`, `[PSEUDOCODE]` comment markers from the LaTeX source
  files before including them in the submission (they're harmless in the PDF
  but look odd in the source).

- [ ] README.md: read through for any AI-flavoured phrasing or instructions.

- [ ] `git status` → commit all outstanding changes with a clean commit message.
- [ ] `uv run pytest` → confirm all tests pass.
- [ ] Confirm `main` is the submission branch.
- [ ] `git tag submission && git push origin submission`
- [ ] Confirm repo is public or accessible to Chris.

### 4. SUBMIT

- [ ] Submit the PDF and/or repo link per the submission instructions.

---

## THINGS THAT ARE NOT WORTH DOING NOW

- White space / float placement issues — minor cosmetic, no mark impact
- Adding beer1996 citation — bib entry was removed; not worth reinstating
- Further content changes — word count is 2575/3000, report is complete

---

## WORD COUNT TOOL

To count the body text (excludes methods, captions, bibliography):

```bash
cd report/
texcount -sum -1 sections/introduction.tex sections/results.tex \
         sections/discussion.tex report.tex 2>/dev/null
```

Or approximately, count only the four body sections' prose from the source.
The current count (verified): abstract 134 + intro 615 + results 887 +
discussion 939 = 2575 words.
