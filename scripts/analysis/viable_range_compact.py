"""Compact viable-range figure for the report.

Loads pre-computed data from figs/viable_range_diagnostics.npz and produces
figs/viable_range_compact.pdf with three panels:
  1. frac_V (training mode) across generations — mean ±1 SD across runs
  2. frac_V (HP-off mode) across generations — assimilation probe
  3. Entry-exit rate at the final generation — grouped bars (training vs HP-off)

Run from repo root:
    uv run python scripts/analysis/viable_range_compact.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_utils import (
    CONDITION_LABELS,
    CONDITION_ORDER,
)

_REPO_ROOT  = Path(os.path.realpath(__file__)).parent.parent.parent
FIGS_DIR    = _REPO_ROOT / "figs"

_PALETTE      = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
_COND_COLOUR  = dict(zip(CONDITION_ORDER, _PALETTE))

MODE_TRAINING = 0
MODE_HP_OFF   = 1


def _frac_v_curves(frac_V: np.ndarray, mode: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-condition (mean, sd) of neuron-averaged frac_V across runs.

    frac_V shape: (n_cond, n_runs, n_gens, n_neurons, n_modes)
    Returns list of (mean, sd), each shape (n_gens,), one per condition.
    SD is across runs (between-run variance), not across neurons.
    """
    neuron_avg = np.nanmean(frac_V[:, :, :, :, mode], axis=3)  # (n_cond, n_runs, n_gens)
    results = []
    for ci in range(neuron_avg.shape[0]):
        arr = neuron_avg[ci]   # (n_runs, n_gens)
        results.append((
            np.nanmean(arr, axis=0),
            np.nanstd(arr, axis=0, ddof=1),
        ))
    return results


def _entry_exit_final(transition_rates: np.ndarray) -> np.ndarray:
    """Mean entry-exit rate at the final sampled generation, per condition per mode.

    transition_rates shape: (n_cond, n_runs, n_gens, n_neurons, n_modes, 3, 3)
    Returns array shape (n_cond, n_modes), means taken over runs and neurons.
    """
    ee = np.nanmean(np.stack([
        transition_rates[..., 1, 0],   # V → U
        transition_rates[..., 1, 2],   # V → O
        transition_rates[..., 0, 1],   # U → V
        transition_rates[..., 2, 1],   # O → V
    ], axis=-1), axis=-1)              # (n_cond, n_runs, n_gens, n_neurons, n_modes)

    fi = ee.shape[2] - 1               # index of final sampled generation
    return np.nanmean(ee[:, :, fi, :, :], axis=(1, 2))  # (n_cond, n_modes)


def main() -> None:
    npz_path = FIGS_DIR / "viable_range_diagnostics.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"NPZ not found: {npz_path}. Run viable_range_diagnostics.py first."
        )

    d     = np.load(npz_path, allow_pickle=False)
    sgi   = d["sampled_gen_indices"]
    frac_V = d["frac_V"]
    trans  = d["transition_rates"]

    curves_train  = _frac_v_curves(frac_V, MODE_TRAINING)
    curves_hp_off = _frac_v_curves(frac_V, MODE_HP_OFF)
    ee_final      = _entry_exit_final(trans)   # (n_cond, n_modes)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    ax1, ax2, ax3 = axes

    labels = [CONDITION_LABELS[c] for c in CONDITION_ORDER]

    # ── Panels 1 & 2: frac_V over generations ─────────────────────────────────
    for ax, curves, title in [
        (ax1, curves_train,  "Viable fraction — training"),
        (ax2, curves_hp_off, "HP-off (assimilation probe)"),
    ]:
        for ci, cond in enumerate(CONDITION_ORDER):
            colour = _COND_COLOUR[cond]
            mean, sd = curves[ci]
            ax.fill_between(sgi, mean - sd, mean + sd, color=colour, alpha=0.18)
            ax.plot(sgi, mean, color=colour, linewidth=1.8, label=CONDITION_LABELS[cond])
        ax.set_xlim(int(sgi[0]), int(sgi[-1]))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Generation", fontsize=10)
        ax.set_ylabel("Viable fraction", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Panel 3: entry-exit rate at final generation — grouped bars ────────────
    bar_width = 0.35
    x = np.arange(len(CONDITION_ORDER))
    cond_colours = [_COND_COLOUR[c] for c in CONDITION_ORDER]

    ax3.bar(
        x - bar_width / 2, ee_final[:, MODE_TRAINING],
        bar_width, label="Training",
        color=cond_colours, alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    ax3.bar(
        x + bar_width / 2, ee_final[:, MODE_HP_OFF],
        bar_width, label="HP-off",
        color=cond_colours, alpha=0.40, edgecolor="white", linewidth=0.5,
        hatch="//",
    )

    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax3.set_ylabel("Entry-exit rate (per 1000 steps)", fontsize=10)
    ax3.set_ylim(bottom=0)
    ax3.set_title("Entry-exit rate (final generation)", fontsize=10, fontweight="bold")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(fontsize=9, frameon=False)

    # ── Shared condition legend from panel 1 ──────────────────────────────────
    handles, lbls = ax1.get_legend_handles_labels()
    fig.legend(
        handles, lbls,
        loc="lower center", ncol=4, fontsize=9,
        frameon=False, bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=[0, 0.10, 1, 1])

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGS_DIR / "viable_range_compact.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
