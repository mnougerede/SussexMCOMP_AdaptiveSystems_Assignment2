# Genotype-based diversity is not computable from the saved data because
# history/ stores population fitnesses but not population genotypes.
# Population fitness spread (per-generation standard deviation of the 30
# fitness values) is used as the diversity proxy; this is a deliberate
# choice, not an oversight.
"""Phase 8d search-dynamics figure.

For each condition, plots the mean across runs of three per-generation
quantities:
  - Population best fitness
  - Population mean fitness
  - Population fitness spread (std of the 30 per-gen fitness values,
    used as a diversity proxy — see comment above)

Saved as figs/search_dynamics_population.pdf.

Run from repo root:
    uv run python scripts/analysis/search_dynamics.py
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

from load_runs import (
    CONDITION_LABELS,
    CONDITION_ORDER,
    population_fitnesses,
    runs_by_condition,
)

FIGS_DIR = Path(os.path.realpath(__file__)).parent.parent.parent / "figs"

_SERIES = ("best", "mean", "spread")
_COLOUR = {"best": "#1f77b4", "mean": "#ff7f0e", "spread": "#2ca02c"}
_STYLE  = {"best": "-",       "mean": "--",      "spread": ":"}
_LW     = {"best": 1.8,       "mean": 1.6,       "spread": 1.8}
_LABEL  = {
    "best":   "Population best",
    "mean":   "Population mean",
    "spread": "Fitness spread (σ)",
}


def _condition_curves(
    runs: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean-across-runs of (best, mean, spread), each shape (n_gens,)."""
    bests: list[np.ndarray] = []
    means: list[np.ndarray] = []
    spreads: list[np.ndarray] = []
    for run in runs:
        pop = population_fitnesses(run)       # (n_gens, pop_size)
        bests.append(pop.max(axis=1))
        means.append(pop.mean(axis=1))
        spreads.append(pop.std(axis=1, ddof=1))
    return (
        np.mean(np.stack(bests),   axis=0),
        np.mean(np.stack(means),   axis=0),
        np.mean(np.stack(spreads), axis=0),
    )


def main() -> None:
    grouped = runs_by_condition()

    counts = {cond: len(grouped[cond]) for cond in CONDITION_ORDER}
    print("Runs per condition:")
    for cond in CONDITION_ORDER:
        print(f"  {CONDITION_LABELS[cond]}: {counts[cond]}")
    if len(set(counts.values())) > 1:
        print("  Note: run counts differ across conditions.")
    print()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(10, 7),
        sharey=True,
        sharex=True,
    )

    legend_handles = None

    for ax, cond in zip(axes.flat, CONDITION_ORDER):
        runs = grouped[cond]
        n = len(runs)

        ax.set_title(CONDITION_LABELS[cond], fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

        if not runs:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="grey", fontsize=10)
            continue

        mean_best, mean_mean, mean_spread = _condition_curves(runs)
        gens = np.arange(len(mean_best))

        handles = []
        for key, arr in (("best", mean_best), ("mean", mean_mean), ("spread", mean_spread)):
            (h,) = ax.plot(
                gens, arr,
                color=_COLOUR[key],
                linestyle=_STYLE[key],
                linewidth=_LW[key],
                label=_LABEL[key],
            )
            handles.append(h)

        ax.text(
            0.97, 0.97,
            f"n={n} run{'s' if n != 1 else ''}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8, color="grey",
        )

        if legend_handles is None:
            legend_handles = handles

    # Axis labels on outer panels only
    for ax in axes[1, :]:
        ax.set_xlabel("Generation", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Fitness", fontsize=10)

    if legend_handles:
        fig.legend(
            legend_handles,
            [_LABEL[k] for k in _SERIES],
            loc="lower center",
            ncol=3,
            fontsize=10,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )

    fig.tight_layout(rect=[0, 0.07, 1, 1])

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGS_DIR / "search_dynamics_population.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
