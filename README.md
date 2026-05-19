# Adaptive Systems Assignment 2 — Homeostatic plasticity in CTRNNs

Replication and extension of Williams (2006), *Homeostatic Adaptive Networks*, Chapter 7.

This project implements a Beer-style ray-sensor agent in pure Python, evolves CTRNN controllers to catch falling circular objects, and compares evolvability across four conditions: no homeostatic plasticity (HP), HP active during a developmental phase before each trial, HP active during each trial, and HP active in both phases. Three analyses are added that Williams did not perform: behavioural trajectory inspection of evolved individuals, per-neuron viable-range diagnostics across evolution, and a frozen-HP test of Stolting, Beer and Izquierdo's (2023) HP-enabled-oscillation hypothesis in Williams' setting.

## Project structure

- `notes/` — Reading notes, design decisions, methods log
- `plan/` — Live planning documents (todo, experiments spec, experiment targets)
- `src/` — Python source code, organised by simulator component
- `tests/` — Unit tests (63 passing)
- `figs/` — Generated figures (gitignored; regenerated from data)
- `experiments/` — Runtime experiment output: checkpoints, history, manifests (gitignored; backed up to OneDrive)
- `batches/` — Batch provenance records (versioned)
- `report/` — LaTeX source and supporting materials
- `scripts/` — Standalone scripts for experiments, profiling, and status reporting

## Setup

Python environment managed with `uv`. Dependencies: `numpy`, `scipy`, `matplotlib`, `tqdm`.

```bash
uv sync                  # create .venv and install dependencies
uv run pytest            # run all tests (63 passing)
```

## Running an experiment

Use `scripts/launch_batch.py` to launch named batches:

```bash
PYTHONPATH=src uv run python scripts/launch_batch.py \
  --batch pilot_01 --conditions no_hp --n_runs 1 --base_seed 42 --n_workers 4
```

Check progress at any time:

```bash
PYTHONPATH=src uv run python scripts/experiment_status.py
```

**Files created per run:**

| Path | Contents |
|---|---|
| `experiments/<batch>/<run>/config.json` | Full `RunConfig` snapshot |
| `experiments/<batch>/<run>/checkpoint.npz` | Latest population, fitnesses |
| `experiments/<batch>/<run>/checkpoint.rng.json` | RNG state for exact resumption |
| `experiments/<batch>/<run>/history/gen_NNNN.npz` | Per-generation fitness stats |
| `experiments/<batch>/<run>/best_per_gen/gen_NNNN.npz` | Best genotype per generation |
| `batches/<label>.json` | Batch provenance record (versioned) |

**Resuming a run:** call `run_experiment` again with the same `output_dir` and a higher `n_gens`. The runner restores RNG state exactly and continues from the last checkpoint. History from a resumed run is element-wise identical to an uninterrupted run with the same seed.

## Key references

- Williams, H. T. P. (2006). *Homeostatic adaptive networks*. PhD thesis, University of Leeds.
- Williams, H. (2005). Homeostatic plasticity improves continuous-time recurrent neural networks as a behavioural substrate. *Proceedings of AMAM 2005*, Ilmenau, Germany.
- Beer, R. D. (1996). Toward the evolution of dynamical neural networks for minimally cognitive behavior. *From Animals to Animats 4*, MIT Press.
- Stolting, L., Beer, R. D. and Izquierdo, E. J. (2023). Characterizing the role of homeostatic plasticity in central pattern generators. *Proceedings of ALIFE 2023*, MIT Press.

## Author

Max Nougerede, University of Sussex, MComp Adaptive Systems 2025/26.
