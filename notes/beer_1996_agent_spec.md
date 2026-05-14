# Beer 1996 / Williams 2006 Chapter 7 — Agent specification

Implementation reference for the ray-sensor agent. Pulls together the spec from Beer's 1996 *Animats 4* paper and Williams' 2006 thesis Chapter 7 into a single document, so we don't have to keep going back to either source while coding.

Where Beer and Williams differ, **Williams takes precedence** because we're replicating his experiments, not Beer's. Notable differences are flagged.

---

## Environment

- 2D plane, dimensions 400 × 275 (Beer 1996). Williams does not specify dimensions explicitly — adopt Beer's.
- Coordinate origin: typically bottom-left, with x horizontal and y vertical, but the specific choice doesn't matter as long as it's consistent.
- Gravity acts downward (negative y direction) on falling shapes.

---

## Agent body

- Circular, radius 5 (Williams 2006 §7.4.1).
- Constrained to horizontal motion only ("as if mounted on a frictionless rail", Williams 2006 §7.4.1.3).
- Zero mass — no momentum or inertia.
- Position $x$ updates per timestep according to the motor equation below.

---

## Sensors

Three ray sensors mounted on the upper periphery of the agent. The array spans an arc of $\pi/6$ radians ($30°$), upward-facing, with sensors evenly distributed across that arc (Williams 2006 §7.4.1.2).

Each sensor is a ray extending from the agent's surface (not its centre) upward into the environment. The ray is infinite-length in principle but only intersections within $D_{max}$ produce a sensor reading.

When a ray intersects a shape, the sensor returns:

$$S = S_{max} \cdot \frac{D_{max} - D}{D_{max}}$$

where:
- $S$ is the sensor signal magnitude returned to the corresponding CTRNN sensor neuron as external input $I$
- $S_{max} = 5$ (Williams 2006)
- $D_{max} = 100$ (Williams 2006)
- $D$ is the distance from the ray's origin (on the agent's body) to the intersection point

If no intersection occurs within $D_{max}$, the sensor returns 0.

**Implementation note:** if multiple shapes are present and a ray could intersect more than one, return the signal corresponding to the *closest* intersection. In Williams' setup only one shape is in flight at a time, so this case doesn't arise, but it's worth handling defensively.

---

## Motors

The agent has two motors acting in opposite directions. The output of the two motor neurons drives the agent's horizontal motion via:

$$\tau_x \dot{x} = z_{right} - z_{left}$$

where:
- $z_{left}$ and $z_{right}$ are the firing rates of the two motor neurons
- $\tau_x = 0.2$ (Williams 2006)

For convenience, Williams chooses $\tau_x$ to equal the integration timestep, so the per-step change in $x$ is just $z_{right} - z_{left}$.

---

## Control network

5-node fully connected CTRNN with no interneurons (Williams 2006 §7.4.1.1):
- 3 nodes are sensor neurons, each receiving external input from one of the three ray sensors
- 2 nodes are motor neurons, whose firing rates drive the motor equation

All 5 nodes are connected to all 5 nodes (including self-connections).

Network state equation:

$$\tau_i \dot{y}_i = -y_i + \sum_{j=1}^{5} w_{ji} \, z_j + I_i$$

Activation:

$$z_i = \sigma(y_i + b_i) = \frac{1}{1 + e^{-(y_i + b_i)}}$$

Parameter ranges:
- Weights $w_{ji} \in [-10, 10]$
- Biases $b_i \in [-10, 10]$
- Time constants $\tau_i \in [1, 4]$

Integration: forward Euler with step size 0.2 timesteps.

The input $I_i$ is nonzero only for sensor neurons; for these it carries the ray sensor signal.

---

## Falling shapes

For the ball-catching task (the one we're replicating), shapes are circles of radius 10 (Williams 2006 §7.4.2).

Shape generation per trial:
- Initial horizontal position: randomly chosen from 10 evenly-distributed positions in the range $[x_{agent} - 25, x_{agent} + 25]$, where $x_{agent}$ is the agent's current horizontal position at the moment the shape spawns
- Initial vertical position: 100 units above the agent
- Initial horizontal velocity: 0 (in Williams' ball-catching task; Beer used a wider range)
- Initial vertical velocity: chosen per task; for ball-catching, ${0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5}$ (10 values matching the 10 horizontal offsets — Williams uses random velocities, we can simplify to a fixed set if needed)

The shape disappears when its lowest point passes the highest point of the agent (i.e. when the shape's leading edge has reached the agent's altitude).

**Note** — Williams' falling shape moves down at constant velocity (no acceleration). The "gravity" in the environment is purely kinematic and the velocity remains constant for the lifetime of a shape.

---

## Trial structure

Williams (2006 §7.4.3.4):
- Each fitness evaluation consists of multiple ball-catching trials
- In each trial, a single shape is dropped and the agent has from its appearance until its leading edge reaches the agent's level to catch it
- "Catching" means: the agent's body overlaps horizontally with the shape's centre when the shape's leading edge meets the agent's leading edge
- Distance $d_i$ for the $i$-th trial: horizontal distance between agent centre and shape centre at the moment leading edges meet, normalised by some maximum

Fitness function (Williams 2006 equation 7.3):

$$F = \frac{1}{N_{trials}} \sum_i \left(1 - \frac{d_i}{d_{max}}\right)$$

normalised to $[0, 1]$. Williams uses $d_{max} = 25$ in the ball-catching task (since horizontal offsets are drawn from $[-25, +25]$, the worst-case horizontal distance is 25 if the agent doesn't move at all).

Number of trials per fitness evaluation: Williams uses 10 evaluation trials per individual per generation, sometimes with additional random-velocity trials.

---

## Simulator architecture

A clean separation we can use:

```
Environment
├── holds: agent, list of falling shapes, environment bounds
├── step(dt): advance simulation by one timestep
└── reset(): clear shapes, reset agent

Agent (body)
├── holds: position, body radius
└── update_position(left_speed, right_speed, dt)

Sensors (separate from body)
├── 3 ray sensors at fixed orientations relative to body
├── compute(shapes, agent_position): return 3 sensor values
└── ray-shape intersection geometry

Controller (CTRNN + optionally HP)
├── holds: 5 neurons with weights, biases, time constants, states
├── step(sensor_inputs, dt): advance CTRNN state, return motor outputs
└── optional HP module that modifies parameters per timestep

Shape
├── position, velocity, type (circle | diamond), size
└── update_position(dt)

Trial
├── shape sequence (pre-generated for reproducibility)
├── timestep loop: get sensors → step controller → step body → step shapes → record state
└── compute fitness from recorded distances
```

The clean separation means HP can be enabled/disabled by swapping in different controller objects without touching the simulator.

---

## Decisions still open during implementation

- **Exact ray geometry:** rays originate from points on the body's circumference, but their angular positions are not fully specified. Williams' figure 7.1 shows a fan with three sensors. We'll evenly distribute three rays across the $\pi/6$ arc, with the middle ray pointing straight up.
- **Trial random seeding:** to make fitness comparisons fair, the same shape sequences should be used across individuals within a single GA evaluation. We seed the trial generator from the generation number, so all individuals in a generation face the same shapes.
- **Per-trial reset of CTRNN state:** Williams doesn't say explicitly. The conservative choice is to reset $y_i = 0$ at the start of each trial within a fitness evaluation, so that fitness measures the controller's reactive capability rather than its state at evaluation start.
- **CTRNN integration step versus trial timestep:** these can be the same (0.2) or the CTRNN can be sub-stepped. Williams uses the same. We use the same.
