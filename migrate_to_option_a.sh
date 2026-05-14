#!/usr/bin/env bash
# migrate_to_option_a.sh — bring a repo bootstrapped from the original Sandbox-oriented
# plan up to date with the Option A decisions: own simulator, Williams Chapter 7
# ball-catching replication, three analyses as the extension.
#
# Run this from the repository root. Idempotent: safe to re-run.
#
# What it does:
#   1. Creates new source directories (sensors/, environment/)
#   2. Rewrites src/agent/README.md (no more "Sandbox glue")
#   3. Rewrites top-level README.md
#   4. Appends new bibliography entries to report/bibliography.bib if missing
#
# What it does NOT do (you handle these manually):
#   - Remove obsolete content in notes/ and plan/ documents (drop in the new versions)
#   - Commit anything

set -e

echo "Migrating repository to Option A structure..."

# New source directories
mkdir -p src/sensors src/environment

# READMEs for the new directories
if [ ! -f src/sensors/README.md ]; then
	cat > src/sensors/README.md <<'EOF'
# Sensors
Ray sensors with ray-shape intersection geometry. Implements Williams (2006) equation 7.1.
EOF
fi

if [ ! -f src/environment/README.md ]; then
	cat > src/environment/README.md <<'EOF'
# Environment
Falling shapes (circles and diamonds) and trial runner. Implements Williams (2006) Chapter 7 experimental setup.
EOF
fi

# Rewrite src/agent/README.md
cat > src/agent/README.md <<'EOF'
# Agent
Agent body, motor mapping, and kinematics. Horizontal motion only.
EOF
echo "  Updated: src/agent/README.md"

# Rewrite src/ctrnn/README.md (was "Own implementation" — make it more specific)
cat > src/ctrnn/README.md <<'EOF'
# CTRNN
Continuous-time recurrent neural network implementation. Implements Williams (2006) equation 3.1.
EOF
echo "  Updated: src/ctrnn/README.md"

# Rewrite top-level README.md
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
echo "  Updated: README.md"

# Bibliography additions
BIB=report/bibliography.bib

add_if_missing() {
	local key="$1"
	local entry="$2"
	if [ -f "$BIB" ] && ! grep -q "^@.*{${key}," "$BIB"; then
		echo "" >> "$BIB"
		echo "$entry" >> "$BIB"
		echo "  Added bibliography entry: $key"
	fi
}

add_if_missing "williams2006" '@phdthesis{williams2006,
	title={Homeostatic Adaptive Networks},
	author={Williams, Hywel Thomas Parker},
	school={University of Leeds},
	year={2006}
}'

add_if_missing "williams_noble_2007" '@article{williams_noble_2007,
	title={Homeostatic plasticity improves signal propagation in continuous-time recurrent neural networks},
	author={Williams, Hywel and Noble, Jason},
	journal={BioSystems},
	volume={87},
	number={2-3},
	pages={252--259},
	year={2007}
}'

add_if_missing "beer1996" '@inproceedings{beer1996,
	title={Toward the evolution of dynamical neural networks for minimally cognitive behavior},
	author={Beer, Randall D.},
	booktitle={From Animals to Animats 4: Proceedings of the Fourth International Conference on Simulation of Adaptive Behavior},
	pages={421--429},
	publisher={MIT Press},
	year={1996}
}'

add_if_missing "stolting2023" '@inproceedings{stolting2023,
	title={Characterizing the role of homeostatic plasticity in central pattern generators},
	author={Stolting, Lindsay and Beer, Randall D. and Izquierdo, Eduardo J.},
	booktitle={Proceedings of the 2023 Artificial Life Conference},
	publisher={MIT Press},
	year={2023},
	doi={10.1162/isal_a_00599}
}'

add_if_missing "mathayomchan_beer_2002" '@article{mathayomchan_beer_2002,
	title={Center-crossing recurrent neural networks for the evolution of rhythmic behavior},
	author={Mathayomchan, Boonyanit and Beer, Randall D.},
	journal={Neural Computation},
	volume={14},
	number={9},
	pages={2043--2051},
	year={2002}
}'

add_if_missing "dipaolo2000" '@inproceedings{dipaolo2000,
	title={Homeostatic adaptation to inversion of the visual field and other sensorimotor disruptions},
	author={Di Paolo, Ezequiel A.},
	booktitle={From Animals to Animats 6: Proceedings of the Sixth International Conference on Simulation of Adaptive Behavior},
	pages={440--449},
	publisher={MIT Press},
	year={2000}
}'

add_if_missing "turrigiano1999" '@article{turrigiano1999,
	title={Homeostatic plasticity in neuronal networks: the more things change, the more they stay the same},
	author={Turrigiano, Gina G.},
	journal={Trends in Neuroscience},
	volume={22},
	number={5},
	pages={221--228},
	year={1999}
}'

echo ""
echo "Migration complete."
echo ""
echo "Next steps:"
echo "  1. Replace plan/todo.md, plan/experiments.md, plan/design_decisions.md, and notes/plasticity_rule.md with the new versions"
echo "  2. Drop the new notes files into notes/: stolting_2023_notes.md, beer_1996_agent_spec.md"
echo "  3. Review the diff: git status; git diff"
echo "  4. Commit: git add -A && git commit -m 'Migrate to Option A: own simulator, ball-catching replication'"
echo "  5. Push"
