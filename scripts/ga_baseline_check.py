"""
GA baseline check (no HP).

Outputs:
  figs/ga_baseline.pdf            – best / mean fitness per generation
  figs/ga_baseline_trajectory.pdf – x-position and y-position traces for the
                                    best final individual over one trial

Prints:
  per-generation best / mean fitness
  runtime per fitness evaluation (mean ± std, measured over generation 0)
  ETA after generation 0
  final best fitness
"""

import multiprocessing
import os
import sys
import time
from pathlib import Path
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt

from agent.body import AgentBody
from ctrnn.agent import CTRNNAgent
from ctrnn.config import CTRNNConfig
from environment.fitness import evaluate_fitness
from environment.trial import run_trial
from ga.ga import GA
from plasticity.hp import HP

def _eval_worker_baseline(genotype: np.ndarray, worker_seed: int) -> float:
    """Top-level so it is picklable by multiprocessing.Pool."""
    rng = np.random.default_rng(worker_seed)
    agent = CTRNNAgent(ctrnn_config)
    agent.load_genotype(genotype)
    return evaluate_fitness(agent, HP(), rng, n_trials=N_TRIALS, n_shapes=N_SHAPES, hp_mode=HP_MODE)


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


# ── Run parameters ────────────────────────────────────────────────────────────
POP_SIZE  = 30
N_GENS    = 100
N_TRIALS  = 5
N_SHAPES  = 20
SEED      = 42
HP_MODE   = 'none'
N_WORKERS = 6

ROOT     = Path(__file__).parent.parent
FIGS_DIR = ROOT / 'figs'
FIGS_DIR.mkdir(exist_ok=True)

# ── Initialise GA ─────────────────────────────────────────────────────────────
rng          = np.random.default_rng(SEED)
ctrnn_config = CTRNNConfig()
ga           = GA(
    pop_size=POP_SIZE,
    genotype_length=ctrnn_config.genotype_length,
    rng=rng,
)
population = ga.initial_population()

best_history = []
mean_history = []
eval_times_gen0: list[float] = []

# ── Evolution loop ────────────────────────────────────────────────────────────
t_run_start = time.monotonic()

for gen in range(N_GENS):
    fitnesses = np.empty(POP_SIZE)
    gen_times: list[float] = []

    if N_WORKERS == 1:
        for i in range(POP_SIZE):
            agent = CTRNNAgent(ctrnn_config)
            agent.load_genotype(population[i])
            hp = HP()

            t0 = time.monotonic()
            fitnesses[i] = evaluate_fitness(
                agent, hp, rng,
                n_trials=N_TRIALS, n_shapes=N_SHAPES, hp_mode=HP_MODE,
            )
            gen_times.append(time.monotonic() - t0)
    else:
        worker_args = [
            (population[i], SEED * 10000 + gen * 1000 + i)
            for i in range(POP_SIZE)
        ]
        t0 = time.monotonic()
        with multiprocessing.Pool(N_WORKERS) as pool:
            fitnesses[:] = pool.starmap(_eval_worker_baseline, worker_args)
        gen_times.append(time.monotonic() - t0)

    best_history.append(float(fitnesses.max()))
    mean_history.append(float(fitnesses.mean()))

    if gen == 0:
        eval_times_gen0 = gen_times
        mean_t = np.mean(eval_times_gen0)
        std_t  = np.std(eval_times_gen0)
        if N_WORKERS == 1:
            eta_s = mean_t * POP_SIZE * (N_GENS - 1)
            print(f"Runtime per eval (gen 0): mean={mean_t:.3f}s  std={std_t:.3f}s")
        else:
            eta_s = mean_t * (N_GENS - 1)
            print(f"Runtime for gen 0 ({N_WORKERS} workers): {mean_t:.1f}s")
        print(f"Estimated remaining time: {eta_s / 60:.1f} min\n")

    elapsed = time.monotonic() - t_run_start
    mean_gen_s = elapsed / (gen + 1)
    eta_rolling = mean_gen_s * (N_GENS - gen - 1)
    w = len(str(N_GENS))
    print(f"Gen {gen:0{w}d}/{N_GENS}  best={best_history[-1]:.4f}  mean={mean_history[-1]:.4f}"
          f"  elapsed={_fmt_duration(elapsed)}  ETA={_fmt_duration(eta_rolling)}")

    if gen < N_GENS - 1:
        population = ga.step(population, fitnesses)

total_s = time.monotonic() - t_run_start
print(f"\nTotal runtime: {total_s / 60:.1f} min")
print(f"Final best fitness: {best_history[-1]:.4f}")

# ── Figure 1: fitness curves ──────────────────────────────────────────────────
gens_ax = np.arange(N_GENS)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(gens_ax, best_history, label='Best fitness',  linewidth=1.5, color='steelblue')
ax.plot(gens_ax, mean_history, label='Mean fitness',  linewidth=1.5, color='darkorange',
        linestyle='--')
ax.axhline(0.5, color='grey', linestyle=':', linewidth=1, label='~Random baseline')
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')
ax.set_ylim(0, 1)
ax.legend()
ax.set_title(f'GA baseline – no HP  (pop={POP_SIZE}, gens={N_GENS}, trials={N_TRIALS})')
fig.tight_layout()
out1 = FIGS_DIR / 'ga_baseline.pdf'
fig.savefig(out1)
plt.close(fig)
print(f"Saved {out1}")

# ── Best individual trajectory ────────────────────────────────────────────────
best_idx      = int(np.argmax(fitnesses))
best_genotype = population[best_idx]

best_agent = CTRNNAgent(ctrnn_config)
best_agent.load_genotype(best_genotype)
best_body  = AgentBody()
# Separate seed so trajectory is independent of the evolutionary rng stream
traj_rng   = np.random.default_rng(SEED + 999)
record = run_trial(
    best_agent, HP(), best_body, traj_rng,
    n_shapes=N_SHAPES, hp_mode=HP_MODE,
)

# ── Figure 2: trajectory ──────────────────────────────────────────────────────
fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

# Top panel: agent x and shape x vs concatenated time
t_off = 0
episode_boundaries: list[int] = []
for i, (bx, sx) in enumerate(zip(record.body_xs, record.shape_xs)):
    t = np.arange(len(bx)) + t_off
    ax_x.plot(t, sx, color='silver',    linewidth=0.9, alpha=0.9)
    ax_x.plot(t, bx, color='steelblue', linewidth=0.9, alpha=0.9)
    if t_off > 0:
        episode_boundaries.append(t_off)
    t_off += len(bx)

for b in episode_boundaries:
    ax_x.axvline(b, color='lightgrey', linewidth=0.5, zorder=0)

ax_x.legend(
    [Line2D([0], [0], color='steelblue'), Line2D([0], [0], color='silver')],
    ['Agent x', 'Shape x'],
    loc='upper right', fontsize=8,
)
ax_x.set_ylabel('x position')
ax_x.set_title(
    f'Best individual (gen-{N_GENS - 1} fitness={best_history[-1]:.4f}) '
    f'– x-position over one trial  ({N_SHAPES} shapes)'
)

# Bottom panel: shape y-position (shows fall speed/duration per episode)
t_off = 0
for i, sy in enumerate(record.shape_ys):
    t = np.arange(len(sy)) + t_off
    ax_y.plot(t, sy, color='darkorange', linewidth=0.8, alpha=0.6)
    if t_off > 0:
        ax_y.axvline(t_off, color='lightgrey', linewidth=0.5, zorder=0)
    t_off += len(sy)

ax_y.set_xlabel('Timestep')
ax_y.set_ylabel('Shape y position')
ax_y.set_title('Shape y-position (fall trajectory per episode)')

fig.tight_layout()
out2 = FIGS_DIR / 'ga_baseline_trajectory.pdf'
fig.savefig(out2)
plt.close(fig)
print(f"Saved {out2}")
