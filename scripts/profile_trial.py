"""
Profile a single run_trial call to diagnose fitness-evaluation cost.

Prints:
  - total trial wall time
  - estimated time per timestep
  - estimated time per fitness evaluation (5 trials)
  - estimated time for the full experiment (4 conditions × 5 runs × 300 gens × 30 individuals)
  - cProfile top-20 cumulative lines
"""

import cProfile
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from agent.body import AgentBody
from ctrnn.agent import CTRNNAgent
from ctrnn.config import CTRNNConfig
from environment.trial import run_trial
from plasticity.hp import HP

SEED     = 42
N_SHAPES = 20
HP_MODE  = 'none'

rng          = np.random.default_rng(SEED)
ctrnn_config = CTRNNConfig()
agent        = CTRNNAgent(ctrnn_config)
agent.load_genotype(rng.uniform(-1, 1, ctrnn_config.genotype_length))
hp           = HP()
body         = AgentBody()

# ── Warm-up (import caches, JIT) ──────────────────────────────────────────────
_ = run_trial(agent, hp, AgentBody(), np.random.default_rng(0),
              n_shapes=2, hp_mode=HP_MODE)

# ── Timed run ─────────────────────────────────────────────────────────────────
body2 = AgentBody()
rng2  = np.random.default_rng(SEED + 1)
agent.load_genotype(agent.genotype)   # reset weights from genotype

t0     = time.perf_counter()
record = run_trial(agent, hp, body2, rng2, n_shapes=N_SHAPES, hp_mode=HP_MODE)
t_wall = time.perf_counter() - t0

total_steps = sum(len(bx) for bx in record.body_xs)

print("=" * 60)
print(f"Total trial time        : {t_wall:.4f} s")
print(f"Total timesteps         : {total_steps}")
print(f"Time per timestep       : {t_wall / total_steps * 1e6:.2f} µs")
print(f"Est. 5-trial eval       : {t_wall * 5:.2f} s")
n_full = 4 * 5 * 300 * 30
print(f"Est. full experiment    : {t_wall * 5 * n_full / 3600:.1f} h  "
      f"({n_full} evals × 5 trials)")
print("=" * 60)

# ── cProfile ─────────────────────────────────────────────────────────────────
body3 = AgentBody()
rng3  = np.random.default_rng(SEED + 2)
agent.load_genotype(agent.genotype)

pr = cProfile.Profile()
pr.enable()
run_trial(agent, hp, body3, rng3, n_shapes=N_SHAPES, hp_mode=HP_MODE)
pr.disable()

print("\ncProfile – top 20 by cumulative time:")
print("-" * 60)
stats = pstats.Stats(pr, stream=sys.stdout)
stats.sort_stats('cumulative')
stats.print_stats(20)
