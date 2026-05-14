# Experiments module — status

## Built (Passes 1–3)

### Pass 1 — Config layer

`src/experiments/config.py` and per-module `config.py` files in `src/ctrnn/`,
`src/plasticity/`, `src/environment/`, `src/agent/`, `src/ga/`.

- `CTRNNConfig`, `HPConfig`, `EnvConfig`, `AgentConfig`, `GAConfig` — all defaults
  from Williams (2006) Chapter 7.
- `Condition` enum: `HP_OFF`, `HP_DEV_ONLY`, `HP_BEHAVIOUR_ONLY`, `HP_BOTH`.
- `RunConfig` — composes all sub-configs, condition, seed, output dir, dev phase steps.
- `run_config_to_json` / `run_config_from_json` — full round-trip; enum serialises as
  name string; nested dataclasses reconstruct as typed instances.

### Pass 2 — IO / persistence layer

`src/experiments/io.py`

- **Atomic writes**: `atomic_write_json`, `atomic_write_npz` — every file write goes
  via a `.tmp` sibling then `os.replace`, so partial writes never corrupt live files.
- **Manifest**: `load_manifest`, `save_manifest`, `register_run` (idempotent on
  run_id), `update_run_entry`.
- **Checkpoint**: `save_checkpoint` / `load_checkpoint` — numpy arrays in
  `checkpoint.npz`, numpy Generator state dict in `checkpoint.rng.json` sidecar
  (npz cannot hold nested dicts natively). Returns `None` if either file is absent.
- **Per-generation history**: `append_generation_record` writes `history/gen_NNNN.npz`;
  `load_history` loads all files and stacks along a generation axis.
- **Best per generation**: same pattern under `best_per_gen/`.
- `get_git_commit` — current HEAD hash or empty string.

### Pass 3 — Stub runner

`src/experiments/evolve.py`

- `run_experiment(run_config, manifest_path)`:
  1. Creates output dir.
  2. First run: stamps `git_commit`, writes `config.json`.
  3. Resume: checks saved vs current git commit; raises `RuntimeError` if both
     non-empty and mismatched.
  4. Registers/updates manifest entry as `in_progress`.
  5. Loads checkpoint if present; restores RNG and `start_gen`. Otherwise seeds
     from `run_config.seed`.
  6. Loop: stub `rng.random()` population and fitnesses, checkpoint, history,
     best-per-gen, manifest `current_gen`.
  7. Marks manifest `complete`.
- `__main__` block: 20-generation smoke test at `results/data/test_run/`.

## Not yet built

| Module | What's needed |
|---|---|
| `src/ctrnn/` | CTRNN class: state equation, Euler step, weight/bias/tau encoding |
| `src/plasticity/` | HP module: `ρ(z)`, synaptic scaling, intrinsic plasticity |
| `src/sensors/` | Ray-circle intersection, distance-to-signal conversion |
| `src/agent/` | Horizontal kinematics, ray fan placement |
| `src/environment/` | Shape generation, trial loop, fitness function (Williams eq. 7.3) |
| `src/ga/` | Elitist selection, two-type mutation, genotype ↔ phenotype mapping |
| `src/viz/` | Trajectory plots, neural-state subplots, per-neuron viability diagnostics |

## Wiring remaining

Replace the two stub `rng.random()` calls in `evolve.py` (population, fitnesses) with
real evaluation:
1. Decode genotype → CTRNN parameters.
2. Optionally run the developmental HP phase.
3. Run fitness trials using the environment/trial runner.
4. Optionally keep HP active during trials.
5. Apply GA selection and mutation to produce the next population.

## Where to pick up next

1. `src/ctrnn/` — CTRNN class, unit-tested against known equilibria.
2. `src/plasticity/` — HP module, unit-tested against the sign-of-ρ cases.
3. `src/sensors/` + `src/agent/` — geometry, unit-tested with known intersection cases.
4. `src/environment/` — trial runner + fitness, validated with a hand-coded controller.
5. `src/ga/` — real GA, validated with a single-condition shakedown run.
6. Wire into `evolve.py`; confirm fitness curves rise from generation 0.
