# Design decisions — plan-level notes

See `notes/design_decisions.md` for the full methodological record (simulator choice,
task selection, CTRNN architecture, HP parameters, GA scaling, etc.).

---

## Infrastructure order: persistence layer before CTRNN (Passes 1–3)

**Decision:** Build config dataclasses, IO/persistence layer, and the stub experiment
runner before any science code (CTRNN, HP, sensors, GA).

**Reasoning:** The persistence layer touches everything else. Every experiment run
writes checkpoints, history records, and manifest entries regardless of what the GA
or CTRNN does. Building IO first means:

- The resumption guarantee (RNG state faithfully saved and restored) is tested in
  isolation, not entangled with CTRNN correctness.
- Science modules can be dropped in one at a time and immediately benefit from atomic
  writes, git-mismatch guards, and checkpoint-based resumption.
- The stub runner (`rng.random()` population and fitnesses) makes it possible to run
  the full infrastructure end-to-end and verify the manifest and history files are
  well-formed before the science code exists.

**Trade-off:** the stub runner produces no meaningful fitness data. This is intentional
— it is a scaffold, not a result. Replace the two stub calls in `evolve.py` once the
CTRNN and environment are wired.
