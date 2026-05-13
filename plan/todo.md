# Assignment 2 — CTRNN Homeostatic Plasticity Project: To-Do List

**Deadline:** 28 May 2026. Two other assignments running in parallel. Roughly two weeks available.

This is the live to-do list. Re-check it at the start of each working session.

---

## Phase 0 — Grounding (mostly done)

- [x] Read Williams 2005 end-to-end, with notes on equations, agent setup, developmental phase protocol
- [x] Work through the plasticity rule with sign checks
- [x] Map Williams' setup onto Ashby's essential variables framing
- [x] Synthesise reading notes into reference documents
- [ ] Get Sandbox running with the basic CTRNN demo
	- [ ] Download Sandbox from Canvas, set up folder structure
	- [ ] Set up conda environment using `AS2026.yaml` (or equivalent for Ubuntu)
	- [ ] Run the basic Sandbox demo — confirm visualisation works
	- [ ] Open the `stochsearch_and_CTRNN.zip` demo and read code before running
	- [ ] Run it; observe what a CTRNN-controlled agent looks like in motion
- [ ] Sense-check: does the project still feel coherent given what's been learned?
- [ ] Upload `williams2007homeostatic.pdf` to project and skim for any methodological clarifications

---

## Phase 1 — Set up the assignment 2 project

- [ ] Create GitHub repo `SussexMCOMP_AdaptiveSystems_Assignment2`
	- [ ] Initialise with README, Python `.gitignore`, no license (or MIT)
- [ ] Clone locally into IntelliJ workspace
- [ ] Run `bootstrap.sh` to create directory structure
- [ ] Drop the notes documents (`williams_2005_notes.md`, `plasticity_rule.md`, `design_decisions.md`) into `notes/`
- [ ] Drop this to-do list and the experiments-design document into `plan/`
- [ ] First commit and push

---

## Phase 2 — Finalise and submit proposal

- [ ] Re-read the existing proposal draft with new understanding
- [ ] Verify references that have "to be verified" notes:
	- [ ] Williams 2005 venue/citation details (AMAM 2005, Ilmenau — confirmed)
	- [ ] Di Paolo adaptivity definition source (check autopoiesis lecture for citation)
- [ ] Rewrite proposal to reflect:
	- [ ] Moving-light phototaxis as task (with justification)
	- [ ] Naturalistic stationary input during development (with justification)
	- [ ] Evolvability as primary measure
	- [ ] Primary extension: developmental phase duration sweep
	- [ ] Possible secondary extensions: $\tau_w, \tau_b$ variation; alternative $\rho$ shape; development input regime as additional axis
- [ ] Count abstract word count; trim if over 150
- [ ] Transfer content into `proposal_form.docx`
- [ ] Email proposal to Chris with brief covering note explaining the changes from initial sketch

---

## Phase 3 — Baseline implementation

- [ ] Write own CTRNN class
	- [ ] State and activation equations
	- [ ] Euler integration with step 0.2
	- [ ] Parameters: weights, biases, time constants within Williams' ranges
	- [ ] Unit tests against `madvn/CTRNN` reference
- [ ] Wire CTRNN into Sandbox as agent controller
	- [ ] Single light source; two light sensors; differential drive motors
	- [ ] 5-neuron fully connected network: 2 sensor, 1 interneuron, 2 motor
	- [ ] Random initialisation of weights, biases, time constants
- [ ] Run baseline phototaxis trial; visualise behaviour
- [ ] Measure neuron output distributions for a randomly initialised network
	- [ ] Plot histogram of $z$ values across all neurons over a trial
	- [ ] Verify many neurons are saturated near 0 or 1

**Gate:** the saturation problem is visible in the data. If not, debug.

---

## Phase 4 — Homeostatic plasticity implementation

- [ ] Implement the Williams plasticity rule as a standalone Python function (see `plasticity_rule.md` for pseudocode and unit tests)
- [ ] Unit-test the rule with the table of cases
- [ ] Integrate as a developmental phase: HP runs for N timesteps with sensory input from environment but motors disabled, then HP is frozen
- [ ] Add real-time visualisation of neuron states with $H_L, H_U$ overlaid
- [ ] Verify post-development neuron output distributions are shifted into the viable range

**Gate:** developmental HP produces a network with most neurons in the responsive region.

---

## Phase 5 — Add the GA loop

- [ ] Implement (or wire in) GA: population 30, 300 generations, elitism + point mutation
- [ ] Encode CTRNN parameters as real-valued genotype
- [ ] Fitness function: mean negative distance to light over moving-light trial, normalised to $[0, 1]$
- [ ] Run a single GA on baseline (no HP) condition, confirm it produces a fitness improvement curve
- [ ] Save raw data per run (genotypes, fitnesses) for re-analysis

**Gate:** evolution works on the baseline condition and fitness curves rise from generation 0.

---

## Phase 6 — Williams replication (Milestone)

- [ ] Run the three Williams conditions, 5 evolutionary runs each:
	- [ ] No HP at all
	- [ ] HP during fixed-duration development, frozen for evolution
	- [ ] HP active throughout evolution and during behaviour
- [ ] Plot mean best-fitness-per-generation curve with error bars across runs, per condition
- [ ] Verify the qualitative finding: developmental HP outperforms baseline and online HP

**Gate:** Williams' core qualitative result reproduced. This is the headline replication regardless of what comes next.

---

## Phase 7 — Primary extension: developmental duration sweep (Milestone)

- [ ] Choose duration values to sweep: suggest $\{0, 500, 1500, 3000, 6000, 12000\}$ timesteps
- [ ] For each duration: 5 evolutionary runs of full GA, with fixed HP during development then frozen
- [ ] Plot fitness-at-final-generation vs duration, with error bars
- [ ] Plot fitness curves for each duration on same axes
- [ ] Analyse shape: is the relationship monotonic, plateau, threshold, or non-monotonic?

**Gate:** quantitative characterisation of developmental duration dependency.

---

## Phase 8 — Secondary extensions (if time permits)

Pick at most one. Most likely to be cut if time-poor.

- [ ] $\tau_w, \tau_b$ sweep with HP-during-behaviour: does faster plasticity help online?
- [ ] Alternative $\rho$ functional forms (Gaussian, smooth sigmoid)
- [ ] Development input regime as additional axis (zero input vs naturalistic vs random)
- [ ] Equilibrium-detection version of development phase: vary detection threshold instead of fixed duration

---

## Phase 9 — Analysis and writing

- [ ] Neuron-level analyses: time in viable range per neuron, before and after HP, for each condition
- [ ] Failure mode analysis: what do failing GA runs look like?
- [ ] Draft report sections in order: methods → results → analyses → discussion → introduction → abstract
- [ ] Aim for 1-2 main discussion points, build the report toward them
- [ ] Discussion threads: substrate vs evolvability tension; Ashby framing; task-dependence; transient explanation; HP-during-behaviour limitations
- [ ] Final word-count pass

---

## Reading queue (for use during writing, not blocking implementation)

In project files:
- `williams2005homeostatic.pdf` — read
- `williams2007homeostatic.pdf` — to upload and read

To track down during writing:
- Di Paolo (2000) — "Homeostatic adaptation to inversion of the visual field..." — antecedent
- Fine, Di Paolo, Izquierdo (2007) — "Adapting to your body" — for the discussion section
- Beer (1995) — "On the dynamics of small CTRNNs" — canonical CTRNN reference
- Funahashi & Nakamura (1993) — universal approximation result for CTRNNs
- Ashby (1960) *Design for a Brain* — for the discussion framing
- Turrigiano (1999) — biological HP review (Williams' reference [3])

---

## Reference snippets

### CTRNN state equation (from Williams 2005, equation 1)

$$\tau_y \dot{y} = -y + \sum_{i=1}^{N} w_i z_i + I$$

### Activation function (equation 2)

$$z = \sigma(y + b) = \frac{1}{1 + e^{-(y+b)}}$$

### HP rule (equations 3, 4, 5)

$\rho(z) = (H_L - z)/H_L$ if $z < H_L$; $0$ if $H_L \le z \le H_U$; $(H_U - z)/(1 - H_U)$ if $z > H_U$.

$\dot{w} = (1/\tau_w) \rho |w|$, $\dot{b} = (1/\tau_b) \rho$.

With $H_L = 0.25$, $H_U = 0.75$, $\tau_w = 40$, $\tau_b = 20$. Integration step $dt = 0.2$.

### Parameter ranges

$w \in [-10, 10]$, $b \in [-10, 10]$, $\tau_y \in [1, 4]$.

### Environment notes

- CTRNN reference: `github.com/madvn/CTRNN` (numpy 1.26.4 required)
- Sandbox: `AS2026.yaml` conda env
- Working on Ubuntu with IntelliJ + bash terminal

### Key conceptual hooks for the discussion

- HP as self-organisation at the neural level (Ashby ultrastability)
- Neuron output range as essential variable
- Di Paolo's adaptivity definition applied at the network level
- Two timescales of adaptation: developmental vs lifetime
- Substrate-vs-evolvability tension: HP improves substrate properties but evolvability gains are task-dependent
- Online HP's failure as a transient / objective-misalignment problem rather than a fundamental limitation
