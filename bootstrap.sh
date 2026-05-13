#!/usr/bin/env bash
# bootstrap.sh — initialise directory structure for the Assignment 2 project.
# Run this from the root of the cloned GitHub repository.
#
# Usage:
#   chmod +x bootstrap.sh
#   ./bootstrap.sh
#
# Idempotent: safe to re-run; existing files are left alone.

set -e

echo "Creating directory structure..."

# Top-level directories
mkdir -p notes plan report/sections src/ctrnn src/plasticity src/agent src/ga src/experiments src/viz results/figures results/data tests

# Placeholder READMEs in each directory so git tracks empty folders
for dir in notes plan report report/sections src src/ctrnn src/plasticity src/agent src/ga src/experiments src/viz results results/figures results/data tests; do
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
Python modules for the experiment. Subdivided by component." > "$readme" ;;
			src/ctrnn) echo "# CTRNN
Own implementation of the continuous-time recurrent neural network." > "$readme" ;;
			src/plasticity) echo "# Plasticity
Williams homeostatic plasticity rule and variants." > "$readme" ;;
			src/agent) echo "# Agent
Sandbox glue: wires the CTRNN into a Sandbox agent." > "$readme" ;;
			src/ga) echo "# GA
Genetic algorithm for evolving CTRNN parameters." > "$readme" ;;
			src/experiments) echo "# Experiments
Runnable experiment scripts. One file per experiment." > "$readme" ;;
			src/viz) echo "# Visualisation
Plotting code for the standard outputs." > "$readme" ;;
			results) echo "# Results
Saved data and generated figures. Not all committed (see .gitignore)." > "$readme" ;;
			results/figures) echo "# Figures
Generated plots. Committed in final form for the report." > "$readme" ;;
			results/data) echo "# Data
Saved raw experiment data. Not committed by default (see .gitignore)." > "$readme" ;;
			tests) echo "# Tests
Unit tests for the CTRNN, HP rule, and other components." > "$readme" ;;
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

Replication and extension of Williams (2005), *Homeostatic plasticity improves continuous-time recurrent neural networks as a behavioural substrate*.

## Project structure

- `notes/` — Reading notes and reference documents
- `plan/` — Live planning documents (to-do, experiments spec, design decisions)
- `src/` — Python source code
- `tests/` — Unit tests
- `results/` — Generated figures and saved experiment data
- `report/` — LaTeX source and supporting materials

## Setup

Python environment via `uv`. CTRNN reference: `madvn/CTRNN`. Simulator: Sandbox (Adaptive Systems module).

See `plan/todo.md` for current state and next steps.

## References

- Williams, H. (2005). Homeostatic plasticity improves continuous-time recurrent neural networks as a behavioural substrate. *Proceedings of AMAM 2005*, Ilmenau, Germany.

## Author

Max Nougerede, University of Sussex, MComp Adaptive Systems 2025/26.
EOF
fi

# Empty bibliography
if [ ! -f report/bibliography.bib ]; then
	cat > report/bibliography.bib <<'EOF'
@inproceedings{williams2005,
	title={Homeostatic plasticity improves continuous-time recurrent neural networks as a behavioural substrate},
	author={Williams, Hywel},
	booktitle={Proceedings of the International Symposium on Adaptive Motion in Animals and Machines (AMAM 2005)},
	address={Ilmenau, Germany},
	year={2005}
}

@article{beer1995,
	title={On the dynamics of small continuous-time recurrent neural networks},
	author={Beer, Randall D.},
	journal={Adaptive Behavior},
	volume={3},
	pages={469--509},
	year={1995}
}

@book{ashby1960,
	title={Design for a Brain: The Origin of Adaptive Behaviour},
	author={Ashby, W. Ross},
	year={1960},
	publisher={Chapman \& Hall},
	edition={Second}
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
