# Adaptive Systems Assignment 2 — Homeostatic plasticity in CTRNNs

Replication and extension of Williams (2006), *Homeostatic Adaptive Networks*, Chapter 7.

This project implements a Beer-style ray-sensor agent in pure Python, evolves CTRNN controllers to catch falling circular objects, and compares evolvability across four conditions: no homeostatic plasticity (HP), HP active during a developmental phase before each trial, HP active during each trial, and HP active in both phases. Three analyses are added that Williams did not perform: behavioural trajectory inspection of evolved individuals, per-neuron viable-range diagnostics across evolution, and a frozen-HP test of Stolting, Beer and Izquierdo's (2023) HP-enabled-oscillation hypothesis in Williams' setting.

## Project structure

- `notes/` — Reading notes and reference documents
- `plan/` — Live planning documents (to-do, experiments spec, design decisions)
- `src/` — Python source code, organised by simulator component
- `tests/` — Unit tests
- `results/` — Generated figures and saved experiment data
- `report/` — LaTeX source and supporting materials

## Setup

Python environment managed with `uv`. Standard scientific Python only — `numpy`, `scipy`, `matplotlib`, `pandas`, `tqdm`. No third-party simulator or CTRNN library.

See `plan/todo.md` for current state and next steps.

## Key references

- Williams, H. T. P. (2006). *Homeostatic adaptive networks*. PhD thesis, University of Leeds.
- Williams, H. (2005). Homeostatic plasticity improves continuous-time recurrent neural networks as a behavioural substrate. *Proceedings of AMAM 2005*, Ilmenau, Germany.
- Beer, R. D. (1996). Toward the evolution of dynamical neural networks for minimally cognitive behavior. *From Animals to Animats 4*, MIT Press.
- Stolting, L., Beer, R. D. and Izquierdo, E. J. (2023). Characterizing the role of homeostatic plasticity in central pattern generators. *Proceedings of ALIFE 2023*, MIT Press.

## Author

Max Nougerede, University of Sussex, MComp Adaptive Systems 2025/26.
