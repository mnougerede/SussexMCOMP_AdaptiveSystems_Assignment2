#!/usr/bin/env bash
# bootstrap.sh — initialise directory structure for the Assignment 2 project.
# Run this from the root of the cloned GitHub repository.
#
# Usage:
#   chmod +x bootstrap.sh
#   ./bootstrap.sh
#
# Idempotent: safe to re-run; existing files are left alone.
#
# This version reflects the Option A decision: own Python simulator replicating
# Williams (2006) Chapter 7 ball-catching evolvability experiments, with three
# analyses Williams did not include (behavioural trajectories, per-neuron
# diagnostics, and a frozen-HP test of the Stolting et al. 2023 hypothesis).

set -e

echo "Creating directory structure..."

# Top-level directories
mkdir -p notes plan report/sections src/ctrnn src/plasticity src/sensors src/agent src/environment src/ga src/experiments src/viz results/figures results/data tests

# Placeholder READMEs in each directory so git tracks empty folders
for dir in notes plan report report/sections src src/ctrnn src/plasticity src/sensors src/agent src/environment src/ga src/experiments src/viz results results/figures results/data tests; do
	readme="$dir/README.md"
	if [ ! -f "$readme" ]; then
		case "$dir" in
			notes) echo "# Notes
Reference documents for the project. Read these first when picking the project back up after a break." > "$readme" ;;
			plan) echo "# Plan
Live working documents: to-do list, experiments specification, design decisions." > "$readme" ;;
			report) echo "# Report
LaTeX source and supporting materials for the final report." > "$readme" ;;
			report/sections) echo "# Report sections
Per-section drafts. Will be \\\\input{} into main.tex." > "$readme" ;;
			src) echo "# Source code
Python modules for the experiment. Each subdirectory is a single component of the simulator." > "$readme" ;;
			src/ctrnn) echo "# CTRNN
Continuous-time recurrent neural network implementation. Implements Williams (2006) equation 3.1." > "$readme" ;;
			src/plasticity) echo "# Plasticity
Williams' homeostatic plasticity rule (plastic facilitation, synaptic scaling, intrinsic plasticity) operating on a CTRNN instance." > "$readme" ;;
			src/sensors) echo "# Sensors
Ray sensors with ray-shape intersection geometry. Implements Williams (2006) equation 7.1." > "$readme" ;;
			src/agent) echo "# Agent
Agent body, motor mapping, and kinematics. Horizontal motion only." > "$readme" ;;
			src/environment) echo "# Environment
Falling shapes (circles and diamonds) and trial runner. Implements Williams (2006) Chapter 7 experimental setup." > "$readme" ;;
			src/ga) echo "# GA
Genetic algorithm for evolving CTRNN parameters. Simple elitist GA with real-valued genotype and point mutation." > "$readme" ;;
			src/experiments) echo "# Experiments
Runnable experiment scripts. One file per condition or analysis." > "$readme" ;;
			src/viz) echo "# Visualisation
Plotting code for fitness curves, behavioural trajectories, and per-neuron diagnostics." > "$readme" ;;
			results) echo "# Results
Saved data and generated figures. Raw experiment data is not committed (see .gitignore)." > "$readme" ;;
			results/figures) echo "# Figures
Generated plots. Committed in final form for the report." > "$readme" ;;
			results/data) echo "# Data
Saved raw experiment data. Not committed by default (see .gitignore)." > "$readme" ;;
			tests) echo "# Tests
Unit tests for the CTRNN, HP rule, sensors, and other components." > "$readme" ;;
		esac
	fi
done

# .gitignore — Python + scientific Python + LaTeX
if [ ! -f .gitignore ]; then
	cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
.venv/
.env

# Jupyter
.ipynb_checkpoints/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# LaTeX
*.aux
*.log
*.out
*.toc
*.bbl
*.blg
*.synctex.gz
*.fdb_latexmk
*.fls
*.nav
*.snm
*.vrb
*.dvi
*.lof
*.lot

# Experiment outputs (large; do not commit by default)
results/data/*.npz
results/data/*.pkl
results/data/*.h5

# Allow figure outputs and README in results
!results/data/README.md
EOF
fi

# Top-level README
if [ ! -f README.md ]; then
	cat > README.md <<'EOF'
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
EOF
fi

# Bibliography
if [ ! -f report/bibliography.bib ]; then
	cat > report/bibliography.bib <<'EOF'
@phdthesis{williams2006,
	title={Homeostatic Adaptive Networks},
	author={Williams, Hywel Thomas Parker},
	school={University of Leeds},
	year={2006}
}

@inproceedings{williams2005,
	title={Homeostatic plasticity improves continuous-time recurrent neural networks as a behavioural substrate},
	author={Williams, Hywel},
	booktitle={Proceedings of the International Symposium on Adaptive Motion in Animals and Machines (AMAM 2005)},
	address={Ilmenau, Germany},
	year={2005}
}

@article{williams_noble_2007,
	title={Homeostatic plasticity improves signal propagation in continuous-time recurrent neural networks},
	author={Williams, Hywel and Noble, Jason},
	journal={BioSystems},
	volume={87},
	number={2-3},
	pages={252--259},
	year={2007}
}

@inproceedings{beer1996,
	title={Toward the evolution of dynamical neural networks for minimally cognitive behavior},
	author={Beer, Randall D.},
	booktitle={From Animals to Animats 4: Proceedings of the Fourth International Conference on Simulation of Adaptive Behavior},
	pages={421--429},
	publisher={MIT Press},
	year={1996}
}

@article{beer1995,
	title={On the dynamics of small continuous-time recurrent neural networks},
	author={Beer, Randall D.},
	journal={Adaptive Behavior},
	volume={3},
	pages={469--509},
	year={1995}
}

@inproceedings{stolting2023,
	title={Characterizing the role of homeostatic plasticity in central pattern generators},
	author={Stolting, Lindsay and Beer, Randall D. and Izquierdo, Eduardo J.},
	booktitle={Proceedings of the 2023 Artificial Life Conference},
	publisher={MIT Press},
	year={2023},
	doi={10.1162/isal_a_00599}
}

@article{mathayomchan_beer_2002,
	title={Center-crossing recurrent neural networks for the evolution of rhythmic behavior},
	author={Mathayomchan, Boonyanit and Beer, Randall D.},
	journal={Neural Computation},
	volume={14},
	number={9},
	pages={2043--2051},
	year={2002}
}

@inproceedings{dipaolo2000,
	title={Homeostatic adaptation to inversion of the visual field and other sensorimotor disruptions},
	author={Di Paolo, Ezequiel A.},
	booktitle={From Animals to Animats 6: Proceedings of the Sixth International Conference on Simulation of Adaptive Behavior},
	pages={440--449},
	publisher={MIT Press},
	year={2000}
}

@book{ashby1960,
	title={Design for a Brain: The Origin of Adaptive Behaviour},
	author={Ashby, W. Ross},
	year={1960},
	publisher={Chapman \& Hall},
	edition={Second}
}

@article{turrigiano1999,
	title={Homeostatic plasticity in neuronal networks: the more things change, the more they stay the same},
	author={Turrigiano, Gina G.},
	journal={Trends in Neuroscience},
	volume={22},
	number={5},
	pages={221--228},
	year={1999}
}
EOF
fi

echo "Bootstrap complete."
echo
echo "Next steps:"
echo "  1. Drop the notes documents into notes/"
echo "  2. Drop the planning documents into plan/"
echo "  3. git add -A && git commit -m 'Initial project structure'"
echo "  4. git push"
