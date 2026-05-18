"""
Substrate-level sanity check for homeostatic plasticity.

Runs HP on 100 random 5-node CTRNNs and plots firing-rate distributions
before and after adaptation.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt

from ctrnn.agent import CTRNNAgent
from ctrnn.config import CTRNNConfig
from plasticity.hp import HP

N_NETS = 100
N_NODES = 5
RECORD_STEPS = 220
HP_STEPS = 6000
H_L, H_U = 0.2, 0.8

rng = np.random.default_rng(42)
config = CTRNNConfig(n_nodes=N_NODES, dt=0.2)
hp = HP(H_L=H_L, H_U=H_U, tau_w=40, tau_b=20, dt=0.2)
I_zero = np.zeros(N_NODES)

# Each network contributes N_NODES firing-rate samples per recorded timestep.
before_samples = np.empty((N_NETS, RECORD_STEPS, N_NODES))
during_samples = np.empty((N_NETS, RECORD_STEPS, N_NODES))
after_samples  = np.empty((N_NETS, RECORD_STEPS, N_NODES))

for net in range(N_NETS):
    agent = CTRNNAgent(config)
    agent.W = rng.uniform(-10, 10, (N_NODES, N_NODES))
    agent.b = rng.uniform(-10, 10, N_NODES)
    agent.taus = rng.uniform(1, 4, N_NODES)

    # --- record before HP ---
    agent.reset()
    for t in range(RECORD_STEPS):
        agent.step(I_zero)
        before_samples[net, t] = agent.z

    # --- HP training: burn-in then capture final RECORD_STEPS steps ---
    agent.reset()
    for t in range(HP_STEPS - RECORD_STEPS):
        agent.step(I_zero)
        hp.step(agent)
    for t in range(RECORD_STEPS):
        agent.step(I_zero)
        hp.step(agent)
        during_samples[net, t] = agent.z

    # --- record after HP (continue from trained state, no reset) ---
    for t in range(RECORD_STEPS):
        agent.step(I_zero)
        after_samples[net, t] = agent.z

before_flat = before_samples.ravel()
during_flat = during_samples.ravel()
after_flat  = after_samples.ravel()

def frac_outside(arr):
    return np.mean((arr < H_L) | (arr > H_U))

print(f"Fraction outside [{H_L}, {H_U}] before HP:                    {frac_outside(before_flat):.4f}")
print(f"Fraction outside [{H_L}, {H_U}] during HP (final {RECORD_STEPS} steps): {frac_outside(during_flat):.4f}")
print(f"Fraction outside [{H_L}, {H_U}] after  HP:                    {frac_outside(after_flat):.4f}")

# --- figure ---
figs_dir = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(figs_dir, exist_ok=True)

bins = np.linspace(0, 1, 51)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(before_flat, bins=bins, density=True, alpha=0.6, color="silver",    label="before HP")
ax.hist(during_flat, bins=bins, density=True, alpha=0.6, color="darkorange", label=f"during HP (final {RECORD_STEPS} steps)")
ax.hist(after_flat,  bins=bins, density=True, alpha=0.6, color="steelblue",  label="after HP")
ax.axvline(H_L, color="black", linestyle="--", linewidth=1.2, label=f"$H_L = {H_L}$")
ax.axvline(H_U, color="black", linestyle="--", linewidth=1.2, label=f"$H_U = {H_U}$")
ax.set_xlabel("Firing rate")
ax.set_ylabel("Density")
ax.set_title(f"Firing-rate distribution before/after HP  "
             f"({N_NETS} networks × {N_NODES} nodes × {RECORD_STEPS} steps)")
ax.legend()
fig.tight_layout()

out_path = os.path.join(figs_dir, "substrate_check.pdf")
fig.savefig(out_path)
print(f"Figure saved to {out_path}")
