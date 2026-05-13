# Williams (2005) — reading notes

Reference: Williams, H. (2005). *Homeostatic plasticity improves continuous-time recurrent neural networks as a behavioural substrate*. Proceedings of AMAM 2005, Ilmenau, Germany.

These notes synthesise a working understanding of the paper for Assignment 2. They are not a paraphrase — they are organised around the questions and concepts that matter for the project.

---

## 1. The two arguments in the paper

The paper makes two related but distinct arguments. Keep them separate.

**Argument 1 — ensemble-level claims about CTRNNs as a substrate (Figures 5, 6, 7):**
Applied to randomly parameterised CTRNNs, homeostatic plasticity (HP) reliably improves signal propagation and increases the likelihood of autonomous oscillations. These are properties of CTRNNs *as a class of dynamical systems*, independent of any task.

**Argument 2 — task-level claims about evolvability (Figure 8):**
When CTRNNs are embedded in an agent and evolved by a GA on a task, HP's effect on evolvability is *not* uniformly positive. HP applied as a *developmental* mechanism (active for a fixed period, then frozen before behaviour) can help, but the benefit is task-dependent. HP applied *during* behaviour consistently slows evolution.

The headline tension between the two arguments is the conceptual core of the paper: HP produces networks that are "poised to behave" at the substrate level, but this doesn't reliably translate into easier evolution of good controllers at the task level.

---

## 2. The substrate problem: node saturation

CTRNNs use a sigmoidal transfer function $z = \sigma(y+b) = 1/(1+e^{-(y+b)})$ to map neuron potential $y$ to firing rate $z$. The sigmoid has three regions:

- **Saturated low ($z \approx 0$):** the flat tail on the left; changes in $y$ produce negligible changes in $z$.
- **Responsive middle ($z \approx 0.5$):** the steep central region; changes in $y$ produce large changes in $z$.
- **Saturated high ($z \approx 1$):** the flat tail on the right.

A neuron whose habitual potential range puts it in either saturated region is functionally dead: it ignores changes in input and provides constant output. Such neurons cannot oscillate and act as barriers to signal propagation.

For a randomly parameterised CTRNN with weights in $[-10, 10]$, the typical $y$ values are large in magnitude, so most neurons end up saturated. The result: a network whose dynamics are dominated by a few non-saturated neurons, with the rest contributing nothing.

HP is the proposed cure.

---

## 3. Variables in the CTRNN equations

State equation (paper's equation 1):
$$\tau_y \dot{y} = -y + \sum_{i=1}^{N} w_i z_i + I$$

Activation function (paper's equation 2):
$$z = \frac{1}{1+e^{-(y+b)}}$$

For a single neuron, the variables are:

| Symbol | Name | Role | Type |
|---|---|---|---|
| $y$ | Potential | Internal "membrane voltage"; the neuron's integrator state | Dynamic |
| $\dot{y}$ | Rate of change of potential | How $y$ is currently changing | Derived |
| $\tau_y$ | Decay time constant | How fast the neuron's potential responds; range $[1, 4]$ | Fixed (or evolvable) |
| $w_i$ | Synaptic weight from neuron $i$ | Strength and direction (excitatory/inhibitory) of input from neuron $i$; range $[-10, 10]$ | Fixed (or plastic) |
| $z_i$ | Firing rate of upstream neuron $i$ | Input from another neuron | External to this neuron |
| $I$ | External (sensory) input | Drives sensor neurons | Driven by environment |
| $b$ | Bias | Horizontal shift of the sigmoid; intrinsic excitability; range $[-10, 10]$ | Fixed (or plastic) |
| $z$ | Firing rate (output) | $\sigma(y+b)$; what the network and motors see | Derived |

### Reading the state equation

Each term in $\tau_y \dot{y} = -y + \sum_i w_i z_i + I$:

- **$-y$** is the leak term — pulls the potential back toward zero with timescale $\tau_y$. Without other input, the neuron decays to $y = 0$.
- **$\sum_i w_i z_i$** is the summed weighted input from other neurons. Each upstream neuron contributes $w_i z_i$; positive weights are excitatory, negative are inhibitory.
- **$I$** is external sensory input (only nonzero for sensor neurons in Williams' setup).

So a neuron is a leaky integrator with weighted input and external drive. $y$ is what is integrated; $z$ is what is output (after squashing through the sigmoid).

### Reading the activation function

$z = \sigma(y + b)$ takes the potential $y$, shifts it by the bias $b$, and squashes the result through a sigmoid into the range $[0, 1]$. The bias $b$ slides the sigmoid horizontally: a neuron with $b = -5$ needs $y \approx +5$ to fire mid-range; a neuron with $b = +5$ fires mid-range at $y \approx -5$.

The "centre-crossing condition" (Beer 1995, Mathayomchan & Beer 2002) is when each neuron's typical $y$ value lines up with the steepest part of its sigmoid — i.e., when $b$ is set so that the neuron's habitual operating point is in the responsive region. HP pushes the network toward this condition.

### Firing rate — what it means here

In a real neuron, firing rate is spikes per second. In a CTRNN there are no spikes — neurons are continuous-valued. So "firing rate" is just convention for $z$: a number in $[0, 1]$ representing the neuron's activation level. It is the neuron's *output*.

---

## 4. The plasticity rule

### Plastic facilitation $\rho$ (paper's equation 3)

$$\rho = \begin{cases}
(H_L - z) / H_L & 0 \le z < H_L \\
0 & H_L \le z \le H_U \\
(H_U - z) / (1 - H_U) & H_U < z \le 1
\end{cases}$$

With Williams' values $H_L = 0.25$ and $H_U = 0.75$.

Properties:

- $\rho$ is the **homeostatic error signal** — zero when the neuron is in the comfort zone, positive when firing is too low, negative when firing is too high.
- $\rho \in [-1, +1]$ — the function is bounded.
- The function is **piecewise linear** in $z$. Inside the dead zone $[H_L, H_U]$ it is identically zero. Outside, $\rho$ grows linearly with distance from the nearest bound, normalised by the width of the saturation region on that side.
- It is **discontinuous in derivative** at $z = H_L$ and $z = H_U$.
- The bounds are *not* derived — Williams notes in a footnote that the qualitative results are robust to other choices.

### Synaptic scaling rule (paper's equation 4, lecture-slide form)

The paper writes $\tau_w \dot{w} = \rho w$ but explains in prose that the rule "refers to the absolute value of the synaptic weight". The lecture-slide version $\dot{w} = (1/\tau_w) \rho |w|$ is the cleaner statement of the same rule.

Interpretation:
- $|w|$ sets the *magnitude* of the change — larger weights change faster than smaller weights.
- $\rho$ sets the *direction* — positive $\rho$ pushes $|w|$ up, negative $\rho$ pushes $|w|$ down.
- The sign of $w$ is preserved by the rule (an inhibitory connection stays inhibitory, an excitatory one stays excitatory).
- The rule is **non-discriminating between excitatory and inhibitory afferents** — both get pushed in the same direction by $\rho$. When $z$ is too high, all afferents are scaled down in magnitude; when $z$ is too low, all are scaled up.

This works because when all afferent magnitudes shrink toward zero, the neuron's input drive weakens and its $y$ approaches zero, so its $z$ approaches $\sigma(b)$. The bias rule independently maintains $b$ in a reasonable range. The two mechanisms work together.

### Intrinsic plasticity rule (paper's equation 5)

$$\tau_b \dot{b} = \rho$$

The bias slides additively in the direction of $\rho$. When firing is too low ($\rho > 0$), $b$ goes up, the sigmoid shifts left, the neuron becomes more excitable. Simple and clean.

### Timescales

Williams uses $\tau_w = 40$ and $\tau_b = 20$ when both rules are applied simultaneously. These are slow compared to the CTRNN's own timescale ($\tau_y \in [1, 4]$) — HP is a slow drift on top of fast network dynamics.

Williams reports that both rules give similar qualitative results when applied independently, but the published results combine them.

### How plasticity and firing rate relate (the loop)

1. The neuron's firing rate $z$ is determined by CTRNN dynamics at this timestep.
2. The plastic facilitation $\rho$ reads $z$ and outputs an error signal.
3. The plasticity rules use $\rho$ to slowly modify the neuron's *parameters* ($w$ and $b$).
4. On the next timestep, the modified parameters produce a different firing rate (via the CTRNN dynamics).
5. When the firing rate is back in $[H_L, H_U]$, $\rho = 0$ and plasticity stops.

The plasticity acts on parameters, not directly on firing rate. The change in firing rate is a *consequence* of the changed parameters going through the CTRNN dynamics. This is an important conceptual distinction — HP is not clamping the firing rate, it is bending the system's parameters in a direction the dynamics will then carry into a different firing rate.

The mechanism is **local**: each neuron only knows about its own firing rate. HP is therefore a genuinely distributed, self-organising process — exactly the kind of thing the module's framing emphasises.

---

## 5. The connection to Ashby's essential variables

The mapping is direct:

| Ashby's framework | Williams' CTRNN |
|---|---|
| Essential variable | Neuron firing rate $z$ |
| Viable range | $[H_L, H_U] = [0.25, 0.75]$ |
| Out-of-bounds | $z < H_L$ or $z > H_U$ |
| Second-order mechanism | Plasticity rules on $w$ and $b$ |
| "Parameters not state" | HP changes weights/biases, not $y$ directly |

This is exactly the framing the report's discussion should use. It also explains why this topic was a good fit for the module — it is a concrete, working instantiation of Ashby's ultrastability at the neural level.

---

## 6. Network performance metrics (Section III.D)

### Signal propagation

Concretely: for a given network, pick one node as the input node. Hold its input at some value drawn from $[-5, 5]$ for 200 timesteps, then change to a new value drawn from the same range, and measure how much each downstream node's firing rate changes at equilibrium. Repeat 1000 times. Mean $\Delta z$ across nodes is the metric.

Intuition: how much does the output change when the input changes, averaged across the network. High $\Delta z$ means many neurons are in the responsive region and meaningfully coupled to the input.

### Autonomous oscillations

An autonomous oscillation is a regular periodic activity pattern the network maintains *without* needing oscillating input. Some networks settle to a fixed point given constant input; others settle to a limit cycle.

Williams' detection method: a neuron is oscillating if (a) its firing rate has non-zero variance, and (b) the sign of $\dot{y}$ changes regularly and periodically. Both together imply rhythmic firing rate.

The proportion of (network, initial condition) pairs that lead to oscillations is the metric. Williams used 100 initial conditions × 500 random networks of each size.

Why oscillations matter: many behaviours (locomotion, timing, rhythm) need internal temporal structure, which static-equilibrium networks cannot provide. CTRNNs can generate oscillations *if* their parameters are right; HP increases the proportion that do.

---

## 7. The four-condition design (Section IV)

The evolutionary experiments cross two binary axes:

| | No development | With development |
|---|---|---|
| **HP off during behaviour** | Random CTRNN, evolution proceeds directly | HP runs for 6000 timesteps, then frozen, then evolution proceeds |
| **HP on during behaviour** | Random plastic CTRNN, evolution proceeds with HP active | HP runs for 6000 timesteps, then continues during behaviour while evolution proceeds |

Williams' key findings:

- **Best on ball-catching:** fixed CTRNN with development.
- **Best on shape discrimination:** fixed CTRNN *without* development.
- **Worst on both tasks:** plastic CTRNNs (HP active during behaviour), regardless of development phase.

The task-dependence of the development benefit is itself an important finding, and one that argues against simple "HP helps evolvability" conclusions.

### Why HP-during-behaviour hurts (Williams' discussion + speculation)

Williams' own caveats (Section V):

1. Plastic CTRNNs have more state variables ($w$ and $b$ now dynamic), longer transients, and are dynamically more complex than fixed CTRNNs of the same size. The comparison may not be fair.
2. The published view in neuroscience is that HP's function is to counterbalance Hebbian instability, not to improve evolvability per se. Looking for evolvability gains may be the wrong frame.
3. Williams explicitly says "any final conclusions concerning the evolvability of homeostatic plastic networks would be premature".

Plausible additional hypotheses (not in the paper):

4. **Moving-target problem for the GA**: each individual's behaviour depends on where it is in its HP transient. Two genotypes producing identical initial networks can score differently because their plastic adaptation timecourses differ. Fitness signal becomes noisier.
5. **Objective misalignment**: HP regulates to $z \in [H_L, H_U]$. The task may require firing rates near 0 or 1. HP pulls the network away from task-good configurations.
6. **$\rho$'s linear, piecewise functional form**: could be wrong for online use; alternative shapes (smooth, decaying-toward-zero, gated by reward) might do better.

These are good material for the discussion section.

---

## 8. Agent setup (Section III.E) — for reference only

The Beer-style agent: 3 forward-facing ray sensors, 2 motors moving horizontally, fully-connected 5-node CTRNN (each sensor mapped to one node, motor output from two of the remaining nodes), no interneurons. GA: 500 generations, population 50, asexual (elitism + point mutation, no crossover), 10 evolutionary runs per condition, fitness averaged over 10 trials.

We are *not* replicating this exactly. We will use Sandbox with a different (but conceptually equivalent) agent and task.

---

## 9. Key things to walk away with

1. The plasticity rule equations and parameters, with full understanding of what each term does and why.
2. The four-condition design — and which conditions our experiments will engage with.
3. The substrate-vs-evolvability distinction.
4. The Ashby mapping for the discussion.
5. The honest fact that even Williams thinks the comparison is not fully fair, and the multiple plausible reasons why HP-during-behaviour hurts.
6. The task-dependence of the development benefit — and the fact that any single task we use can only speak to one corner of this space.

---

## 10. Things still to check

- **`williams2007homeostatic.pdf`** — not currently in the project files; upload and read for any methodological clarifications.
- **The slide version of the synaptic scaling rule** — confirm with Chris that it matches the paper's intent (it does, but worth confirming).
- **Whether anyone has done quantitative duration sweeps** — a brief literature scan during writing will tell us whether the primary extension is genuinely novel or whether it has been done.
- **Whether anyone has resolved the "HP-during-behaviour is bad" finding** — has more recent work found conditions where online HP does work?
