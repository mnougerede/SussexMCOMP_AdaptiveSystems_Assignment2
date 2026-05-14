# Stolting, Beer & Izquierdo (2023) — reading notes

**Reference:** Stolting, L., Beer, R. D. and Izquierdo, E. J. (2023). Characterizing the role of homeostatic plasticity in central pattern generators. *Proceedings of ALIFE 2023*. MIT Press. DOI: 10.1162/isal_a_00599.

A short conference paper (7 pages) but central to our project's contribution. These notes pull out the structure, the key findings, and the implications for our work.

---

## What the paper does

Stolting, Beer & Izquierdo (SBI) take Williams' HP rule and apply it to a much simpler task than Williams': central pattern generation at a target frequency. The motivation is to understand *why* HP-during-behaviour was found to be detrimental in Williams' agent experiments.

They use the standard Williams formulation:
- CTRNN state equation
- Plastic facilitation $\rho$ piecewise linear in firing rate $z$, with bounds $H_L = 0.25$ and $H_U = 0.75$ (the substrate-experiment values, not Williams Chapter 7's $0.2/0.8$)
- $\tau_w \dot{w} = \rho |w|$, $\tau_\theta \dot{\theta} = \rho$
- $\tau_w = 40$, $\tau_\theta = 20$

The task is: produce a neural oscillation at a target frequency of 0.1 Hz (1 cycle per 10 arbitrary time units). Fitness is $1 - (1/T^2)(x - T)^2$ where $x$ is measured frequency and $T$ is target. Maximum fitness when $x = T$; smooth decline to zero at $x = 0$ or $x = 2T$.

Network is small: $n = 3$ or $n = 2$ in different experiments.

---

## Key finding 1: HP-enabled oscillations

After evolving HP-on circuits to high fitness at the target frequency, they freeze HP on the best individuals and re-test. **In 7 of 50 evolved circuits, freezing HP stopped the oscillation completely.**

This is unexpected on the standard view of HP. The standard view is: HP shifts circuit parameters into a region where oscillations exist, then becomes functionally irrelevant. The frozen test should not destroy the oscillation.

SBI call this phenomenon **HP-enabled oscillation**: the limit cycle exists in the *extended state space* that includes the changing weights and biases. For an $n$-node CTRNN with HP, the dimensionality is not $n$ but $n^2 + 2n$ (n neural states plus $n^2 + n$ changing parameters). Some limit cycles exist only in this extended space.

In contrast, **HP-independent oscillations** are limit cycles that exist in the static-parameter circuit. HP may have pushed the circuit into the region where these exist, but the cycle does not require HP for sustainment.

There is also a third category: **partially HP-enabled** — the homeostatic limit cycle passes through both HP-dependent and HP-independent regions of parameter space. Whether oscillation persists on freezing depends on where HP was when it was frozen.

---

## Key finding 2: HP-enabled oscillations are very common

In a sample of 1000 random 2-node circuits:
- 635 oscillate while HP is on
- Of these, 564 stop oscillating when HP is frozen
- Only ~6% of random circuits showed HP-independent oscillation

So in this small-network setting, HP-enabled oscillation is the dominant mode of HP-driven oscillation, not a rare edge case.

They extend this to larger circuit sizes (n = 1 to 20) and find the pattern holds: HP-on conditions oscillate at much higher rates than no-HP, but most of the gain disappears when HP is frozen.

---

## Key finding 3: timescale separation matters

When they vary the HP timescale $\tau_w = \tau_\theta$ from 1 (same as neural timescale) up to 100 (much slower), they find:
- When HP timescale is close to neural timescale: HP-enabled oscillation is common; many evolved circuits fail when HP is frozen
- As HP timescale gets slower relative to neural timescale: fewer HP-enabled oscillations; more circuits remain oscillating after freezing
- At extreme slow HP (close to "static" relative to behaviour): HP-enabled oscillation becomes uncommon

Williams' canonical values are $\tau_w = 40, \tau_b = 20$, which correspond to "moderate" separation — well within the regime where HP-enabled oscillations occur.

For slower target oscillations (0.05 Hz), HP-enabled oscillation persists at much greater timescale separation than for faster targets.

---

## Their speculation about Williams' result

The discussion includes one paragraph speculating on why HP-during-behaviour hurt Williams' results. The full text (slightly paraphrased):

> Importantly, just because HP-enabled oscillators rely on HP for their dynamics does not inherently mean they are less suitable as a behavioural substrate. However, in the context of the more complex tasks studied by Williams (2004, 2005), HP-enabled oscillations disrupted behavioural fitness. As our analysis elucidates, systems employing HP are rarely ever found at equilibrium points and by definition these equilibria never involve neural saturation. Equilibria and saturation, however, may be crucial to the performance of these tasks. This, in combination with other factors, could be the reason why CTRNNs employing HP during performance are less fit phototaxers, shape discriminators, and ball-catchers.

This is a *speculation*, not a tested claim. They did not actually run their analysis on Williams' agent tasks. They suggest one of two mechanisms:

1. HP-enabled dynamics don't match what the task needs (Williams' tasks may need equilibrium-like neural behaviour, which HP prevents)
2. HP prevents saturation, but saturation may sometimes be useful for these tasks

---

## What this means for our project

The Stolting et al. paper is the perfect anchor for our extension. Their hypothesis is:

> HP-during-behaviour fails on Williams' tasks because HP-enabled dynamics get in the way of behaviour. Freezing HP on evolved HP-during-behaviour individuals should disrupt their behaviour, by the same mechanism that disrupts the HP-enabled oscillators.

This is a specific, testable hypothesis. We test it directly:

1. Run Williams' four-condition replication
2. For each HP-during-behaviour individual at the end of evolution, freeze HP and re-evaluate fitness
3. Look at the fitness drop

If Stolting et al. are right, we should see large fitness drops on freezing. If we don't, their CPG-task finding doesn't transfer to Williams' agent task, and we have to look elsewhere for the explanation.

---

## What we should NOT claim

- We are not testing Stolting et al.'s full framework — they did detailed bifurcation analyses on n=2 networks that we cannot reasonably reproduce.
- We are not claiming HP-enabled oscillation is the same phenomenon in Williams' setting — only that the *signature* (fitness drop on freezing) is the same observable.
- We are not claiming a definitive answer to the question of why HP-during-behaviour fails — we are bringing one specific hypothesis under empirical test.

---

## Citation note

The paper is short and conference-published. The full statement of their hypothesis is the discussion paragraph above. We should quote it (or paraphrase it carefully) when introducing our test.
