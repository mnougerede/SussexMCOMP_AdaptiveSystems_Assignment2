# The Williams plasticity rule

A focused reference document. Open this when implementing or writing about HP.

---

## The three equations

### Plastic facilitation

$$\rho(z) = \begin{cases}
\dfrac{H_L - z}{H_L} & 0 \le z < H_L \\[4pt]
0 & H_L \le z \le H_U \\[4pt]
\dfrac{H_U - z}{1 - H_U} & H_U < z \le 1
\end{cases}$$

With Williams' values: $H_L = 0.25$, $H_U = 0.75$.

### Synaptic scaling

$$\dot{w} = \frac{1}{\tau_w} \rho \, |w|$$

(The paper writes $\tau_w \dot{w} = \rho w$ but specifies the magnitude interpretation in prose. The lecture-slide form above is the cleaner statement of the same rule.)

### Intrinsic plasticity

$$\dot{b} = \frac{1}{\tau_b} \rho$$

### Timescales

$\tau_w = 40$, $\tau_b = 20$ when applied simultaneously.

---

## Sanity checks on the signs

These should be worked through carefully when implementing.

| Condition | $z$ | $\rho$ sign | $\dot{w}$ direction (any $w$) | $\dot{b}$ direction |
|---|---|---|---|---|
| Firing too low | $z < H_L$ | positive | $|w|$ increases (input scaled up) | $b$ increases (shifts sigmoid left → easier to fire) |
| In comfort zone | $H_L \le z \le H_U$ | zero | no change | no change |
| Firing too high | $z > H_U$ | negative | $|w|$ decreases (input scaled down) | $b$ decreases (shifts sigmoid right → harder to fire) |

The rule is **non-discriminating between excitatory and inhibitory afferents** — both get pushed in the same direction by $\rho$. When $z$ is too high, all afferent magnitudes shrink; when $z$ is too low, all grow. The bias rule independently maintains $b$ in a reasonable range, so the two mechanisms work together to bring $z$ back to the comfort zone.

### Worked example: neuron firing too high

Let $z = 0.95$, so $\rho = (0.75 - 0.95) / (1 - 0.75) = -0.8$.

Suppose this neuron has two afferents: $w_1 = +6$ (excitatory) and $w_2 = -4$ (inhibitory).

Per timestep:
- $\dot{w_1} = (1/40) \cdot (-0.8) \cdot |+6| = -0.12$ → $w_1$ shrinks toward zero
- $\dot{w_2} = (1/40) \cdot (-0.8) \cdot |-4| = -0.08$ → $w_2$ also shrinks toward zero (rises since it was negative)

So both magnitudes go down. The neuron receives less total drive (excitatory and inhibitory both weaker). $y$ approaches zero. $z$ approaches $\sigma(b)$.

Simultaneously: $\dot{b} = (1/20) \cdot (-0.8) = -0.04$ → bias shifts negative. The sigmoid moves right. Even if $y$ stays near zero, $z$ now sits below 0.5.

Combined effect: $z$ falls back toward the comfort zone. ✓

---

## Implementation pseudocode

```python
def compute_rho(z, h_l=0.25, h_u=0.75):
    """Plastic facilitation as a function of firing rate."""
    if z < h_l:
        return (h_l - z) / h_l
    elif z > h_u:
        return (h_u - z) / (1.0 - h_u)
    else:
        return 0.0

def apply_homeostatic_plasticity(neuron, dt, tau_w=40.0, tau_b=20.0):
    """One timestep of HP update for a single neuron."""
    rho = compute_rho(neuron.z)
    # Synaptic scaling on all afferent weights
    for i in range(len(neuron.weights)):
        w = neuron.weights[i]
        neuron.weights[i] += dt * (1.0 / tau_w) * rho * abs(w)
    # Intrinsic plasticity on bias
    neuron.bias += dt * (1.0 / tau_b) * rho
```

Notes for our implementation:
- `dt` should be the same integration step used for the CTRNN itself (Williams uses 0.2).
- Apply HP *after* the CTRNN state update for the timestep, not before.
- HP only modifies afferents *into* the neuron whose $z$ is being assessed. Each neuron's afferents are independently regulated by *that neuron's* firing rate, not by upstream neurons' firing rates.

---

## Unit tests to write

| Input | Expected output |
|---|---|
| $z = 0.10$ | $\rho > 0$, bias increases, $|w|$ increases |
| $z = 0.50$ | $\rho = 0$, no parameter change |
| $z = 0.90$ | $\rho < 0$, bias decreases, $|w|$ decreases |
| $z = 0.25$ | $\rho = 0$ (boundary) |
| $z = 0.75$ | $\rho = 0$ (boundary) |
| $z = 0.00$ | $\rho = +1$ (saturated low) |
| $z = 1.00$ | $\rho = -1$ (saturated high) |
| Excitatory ($w > 0$) and inhibitory ($w < 0$) afferents under $\rho < 0$ | both magnitudes shrink |

---

## Open questions worth flagging in writeup

1. The piecewise-linear shape of $\rho$ is one of many possible homeostatic error signals. Alternatives: Gaussian centred at $(H_L + H_U)/2$; smooth sigmoid of distance from target; reward-modulated gating. The functional form is a design choice, not a derivation.
2. The bounds $H_L = 0.25, H_U = 0.75$ are arbitrary (Williams' footnote 1). Some sensitivity testing might be useful in the implementation phase, but not a primary contribution.
3. Williams runs synaptic scaling and intrinsic plasticity simultaneously. Whether either alone is sufficient is worth a brief check during implementation — should be a one-condition addition.
4. Timescales $\tau_w = 40, \tau_b = 20$ are also chosen without explicit justification. The secondary extension (if pursued) could vary these.

---

## Implementation visualisation ideas

These plots should be part of the standard simulation output, not added later:

1. **Histogram of $z$ across all neurons over a trial**, before and after HP development. Should show concentration shifting from tails (0, 1) toward the middle after HP.
2. **Time series of $z$ for each neuron during development**, with $H_L$ and $H_U$ as horizontal lines. Lets us see which neurons enter the viable range, when, and which fail to.
3. **Time series of $w$ and $b$ for each neuron during development**. Should show parameters changing while neurons are out of range, then stabilising once they are in.
4. **Heat map of "time spent in viable range" per neuron, by condition.** Compact summary across conditions.
5. **Overlay of $\rho(z)$ curve with each neuron's current $z$ marked on it**, as a live plot during development. Useful for intuition-building and demos.

Embedding the equations, variable definitions, and current parameter values in the simulation visualisation (as a sidebar or panel) is achievable and useful for demonstration purposes. Worth doing if time permits.
