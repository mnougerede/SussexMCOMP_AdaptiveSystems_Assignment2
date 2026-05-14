# Experiments specification

Concrete plan for the experiments. Updated to reflect Option A. Read alongside `design_decisions.md`.

---

## Experiment 0 — Substrate-level sanity check

**Purpose:** verify that the HP implementation reproduces Williams' substrate-level result (HP shifts random networks away from saturation). Sanity-check, not a main result.

**Setup:** Generate 100 random 5-node fully-connected CTRNNs with parameters drawn from Williams' ranges. For each, run a trial with $I = 0$ and record per-neuron firing rate distribution. Then apply HP for 6000 timesteps. Repeat the trial and record again.

**Output:** Histogram of firing rates across all neurons, before and after HP. Mean fraction of neurons in $[H_L, H_U]$, before and after.

**Expected:** mass shifts from saturated tails (near 0 and near 1) toward the middle. If not, the implementation has a bug.

**Status:** runs in minutes; no compute concern.

---

## Experiment 1 — Williams replication (Williams Chapter 7 Experiments 1 and 2)

**Purpose:** reproduce Williams' four-condition ball-catching evolvability result.

**Conditions:**

| Condition | HP during development phase | HP during fitness trial |
|---|---|---|
| **Baseline** | off | off |
| **Dev-only** | on (6000 timesteps before each trial) | off (HP frozen) |
| **Online** | off | on (HP active throughout trial) |
| **Dev+online** | on (6000 timesteps) | on (HP remains active) |

**Per-condition:** 5 evolutionary runs, different seeds. Population 30, 300 generations.

**Per-individual fitness evaluation:** ~20 ball-catching trials with circles dropped at randomly-chosen horizontal offsets and velocities, as specified in Williams Chapter 7 section 7.4.2. Fitness $= \sum_i (1 - d_i)$ over trials.

**Outputs:**
- Best-fitness-per-generation curves, mean and error band across runs, one per condition (Williams Figure 7.2/7.3 equivalent)
- Final fitness box plots across conditions
- Per-condition mean and standard deviation of final fitness

**Expected:** Williams found that on ball-catching, Dev-only and Baseline reach similar final fitnesses but Dev-only progresses faster in early generations; Online and Dev+online are worse than Baseline. Qualitative reproduction of this ordering is the gate.

**Compute estimate:** 4 conditions × 5 runs × 300 generations × 30 individuals × ~20 trials × ~0.05s/trial ≈ 9 hours total, single-threaded. Halved with 2-way parallel, etc.

**Status:** the headline experiment. Most of the compute budget goes here.

---

## Experiment 2 — Behavioural trajectory analysis

**Purpose:** show what the evolved controllers actually do in each condition. Williams does not include this in the conference paper and only minimally in the thesis.

**Setup:** From each evolutionary run in Experiment 1, take the best individual at the final generation. For each condition, identify three representative individuals (best across runs, median, worst). For each: run a small number of trials (e.g. 5-8) with seeded shape sequences shared across individuals so the comparison is direct.

**Outputs:**
- Per-individual trajectory plot: agent x-position over time, all shape positions, with vertical lines at shape arrival/departure
- Alongside each trajectory plot: per-neuron firing rate over time, with $H_L$ and $H_U$ as horizontal lines
- For each condition: at least one such combined plot, ideally three

**Analysis target:** qualitative comparison of behavioural strategies. Do HP individuals use different neural strategies than non-HP? Are firing rates near saturation, in the viable range, or oscillating?

**Status:** post-hoc analysis of saved data from Experiment 1. Cheap compute.

---

## Experiment 3 — Per-neuron viable-range diagnostics across evolution

**Purpose:** track whether neurons stay in their viable range as evolution proceeds, per condition. Bridges Williams' substrate-level and evolvability claims.

**Setup:** For each condition's evolutionary runs, at each saved generation (every 10 generations is fine for plotting density), take the best individual and run a single representative trial. Record per-neuron fraction of timesteps with firing rate in $[H_L, H_U]$.

**Outputs:**
- Per-condition heat map or line plot: y-axis = neuron index (1 to 5), x-axis = generation, colour or line height = viable-range fraction
- Cross-condition comparison: mean viable-range fraction (averaged across neurons and runs) over generations, one line per condition

**Analysis target:**
- Does the Baseline (no HP) condition find networks where evolution naturally pushes neurons into the viable range, or do successful Baseline controllers tolerate saturation?
- In Online and Dev+online, do firing rates stay in the viable range — or has HP already done its job and evolution can rely on it?

**Status:** post-hoc analysis. Adds modest compute (re-running trials, not re-evolving).

---

## Experiment 4 — Frozen-HP test (the Stolting et al. test)

**Purpose:** test whether HP-during-behaviour controllers rely on continued HP for their behaviour. The project's clearest single scientific question.

**Setup:** For each evolved Online and Dev+online individual at the final generation:
1. Record current fitness with HP active (re-evaluate, not just trust the GA's last fitness; we want a robust measurement with a fresh seeded trial sequence)
2. Freeze the network parameters at their current value
3. Re-evaluate fitness with HP off
4. Record the fitness drop

**Controls:**
- Dev-only individuals already had HP frozen during evolution; re-evaluating these in step 3 is a check on measurement noise
- Baseline individuals never had HP; same check

**Outputs:**
- Box plot of fitness-drop-on-freezing by condition
- Scatter: original fitness vs. frozen fitness, one point per individual, coloured by condition
- Per-individual neural state time-series: with HP and with HP frozen, side by side, for two or three representative cases

**Analysis target:**
- Online individuals: large fitness drop → behaviour was HP-enabled (Stolting et al.'s prediction holds); small drop → HP was incidental, the parameters at the end of evolution were already good
- Dev+online individuals: same logic, with the caveat that they had a developmental head-start
- Magnitude of effect across the condition

**Status:** the contribution's main result. The most important figure for the report.

---

## Run plan

### Week 1
- Day 1: simulator core build (CTRNN, sensors, agent, environment)
- Day 2: simulator validation; HP module; HP unit tests; substrate sanity check (Experiment 0)
- Day 3: GA; single-condition shakedown run
- Day 4: launch Experiment 1; while it runs, write trajectory and diagnostic plotting code
- Day 5: Experiment 1 results in; Experiments 2 and 3 run (post-hoc)

### Week 2
- Day 6: Experiment 4 run
- Days 7–9: writing — methods first, then results and analyses
- Days 10–12: writing — discussion, then introduction, then abstract
- Day 13: figures polish, symbol audit, citation check
- Day 14: final pass, submit

This is tight. Slip on Day 4 by a day and we lose Experiment 4 buffer. Worth running the GA in the background from Day 3 onwards while the rest of the work proceeds.

---

## Compute risk

The four-condition experiment is the compute bottleneck. Mitigations if it's running slow:

- Reduce population from 30 to 20
- Reduce generations from 300 to 200
- Reduce runs per condition from 5 to 3
- Parallelise fitness evaluation across population members

Saving raw data per generation means we can cut the experiment short and still have usable results.

---

## What we'll write up if everything works

The four-condition replication is the table-stakes deliverable: it shows we faithfully reproduced Williams. The three analyses are the contribution. The frozen-HP test, in particular, has a clear pre-registered hypothesis with a binary-ish answer.

## What we'll write up if Experiment 1 doesn't replicate Williams

This is the hardest scenario. Possible reasons: (a) implementation bug, (b) population/generation count too small to see the effect, (c) some methodological difference between our simulator and Williams' that we haven't identified. The reporting strategy then becomes: describe what we found, characterise where it differs from Williams, attempt the three analyses anyway since they don't strictly require Williams' ordering to hold. This is a defensible piece of work even if the headline replication fails.

## What we'll write up if Experiment 4 gives a null result

Stolting et al.'s prediction is that fitness drops on freezing. If it doesn't, that's still a publishable finding — their CPG-task result doesn't transfer to Williams' agent task. The discussion changes shape but the report stays substantive: we have a clear empirical claim and an explanation of why the prediction failed.

Either way, the project has real content. The only failure mode is not finishing the experiments — which is what the run plan is designed to prevent.
