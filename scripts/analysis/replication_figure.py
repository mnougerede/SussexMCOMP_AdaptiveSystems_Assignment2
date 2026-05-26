"""Phase 7 replication figure script.

Produces two publication-quality PDF figures and a statistical summary:
  figs/replication_fitness_curves.pdf  — best fitness per generation, mean ± 1 SD
  figs/replication_final_box.pdf       — final fitness box + jitter by condition

Run from repo root:
    uv run python scripts/analysis/replication_figure.py
"""

import os
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from load_runs import (
    CONDITION_LABELS,
    CONDITION_ORDER,
    best_fitness_series,
    runs_by_condition,
)

EXPECTED_RUNS = 10
_REPO_ROOT = Path(os.path.realpath(__file__)).parent.parent.parent
FIGS_DIR = _REPO_ROOT / "figs"

_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
_COND_COLOUR = dict(zip(CONDITION_ORDER, _PALETTE))


# ─── Data loading and integrity checks ───────────────────────────────────────

def _load_and_validate() -> tuple[dict[str, list[np.ndarray]], dict[str, list[float]]]:
    """Return (curves, finals) dicts keyed by condition.

    curves  — best_fitness series per run (n_gens,)
    finals  — last-generation best_fitness per run
    """
    grouped = runs_by_condition()

    for cond in CONDITION_ORDER:
        n = len(grouped[cond])
        if n != EXPECTED_RUNS:
            print(
                f"WARNING: condition {CONDITION_LABELS[cond]!r} ({cond}) "
                f"has {n} run(s), expected {EXPECTED_RUNS}"
            )

    curves: dict[str, list[np.ndarray]] = {}
    finals: dict[str, list[float]] = {}

    for cond in CONDITION_ORDER:
        curves[cond] = []
        finals[cond] = []
        for run in grouped[cond]:
            series = best_fitness_series(run)
            # best_fitness in history/ is the highest score evaluated in that
            # generation's fresh fitness calls, not a running all-time best.
            # Under stochastic re-evaluation the elitist individual's recorded
            # score varies between generations even though its genotype is
            # preserved, so series[-1] < series.max() is normal and expected.
            curves[cond].append(series)
            finals[cond].append(float(series[-1]))

    return curves, finals


# ─── Figure 1: fitness curves ─────────────────────────────────────────────────

def _fitness_curves_figure(curves: dict[str, list[np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for cond in CONDITION_ORDER:
        label = CONDITION_LABELS[cond]
        colour = _COND_COLOUR[cond]
        series_list = curves[cond]
        if not series_list:
            continue

        matrix = np.stack(series_list)       # (n_runs, n_gens)
        gens = np.arange(matrix.shape[1])
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)

        for row in matrix:
            ax.plot(gens, row, color=colour, alpha=0.15, linewidth=0.7)

        ax.fill_between(gens, mean - sd, mean + sd, color=colour, alpha=0.18)
        ax.plot(gens, mean, color=colour, linewidth=1.8, label=f"{label} (mean ± 1 SD)")

    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Best fitness", fontsize=11)
    ax.set_xlim(0, matrix.shape[1] - 1)
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGS_DIR / "replication_fitness_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─── Figure 2: final fitness box + jitter ────────────────────────────────────

def _final_box_figure(finals: dict[str, list[float]]) -> None:
    labels = [CONDITION_LABELS[c] for c in CONDITION_ORDER]
    data = [finals[c] for c in CONDITION_ORDER]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        boxprops={"linewidth": 1.2},
    )
    for patch, colour in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(colour)
        patch.set_alpha(0.35)

    rng = np.random.default_rng(0)
    for i, (vals, colour) in enumerate(zip(data, _PALETTE), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color=colour,
            s=30,
            zorder=3,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Final best fitness", fontsize=11)
    ax.set_ylim(0.3, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGS_DIR / "replication_final_box.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─── Statistical summary ──────────────────────────────────────────────────────

def _statistical_summary(finals: dict[str, list[float]]) -> None:
    print()
    print("=" * 65)
    print("Statistical summary")
    print("=" * 65)

    # Per-condition descriptives
    print("\nPer-condition final fitness:")
    all_groups: list[list[float]] = []
    for cond in CONDITION_ORDER:
        label = CONDITION_LABELS[cond]
        vals = finals[cond]
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        vals_str = "  ".join(f"{v:.6f}" for v in vals)
        print(f"\n  {label}")
        print(f"    Values : {vals_str}")
        print(f"    Mean   : {mean:.6f}")
        print(f"    Std dev: {sd:.6f}")
        all_groups.append(vals)

    # Kruskal-Wallis
    print()
    print("-" * 65)
    h_stat, p_kw = stats.kruskal(*all_groups)
    print(f"Kruskal-Wallis  H = {h_stat:.4f},  p = {p_kw:.4f}")

    # Pairwise Mann-Whitney U
    print()
    print("-" * 65)
    print("Pairwise Mann-Whitney U (two-sided, Bonferroni correction n=6):")
    print()

    pairs = list(combinations(CONDITION_ORDER, 2))
    n_comparisons = len(pairs)  # 6

    rows = []
    for cond_a, cond_b in pairs:
        a = finals[cond_a]
        b = finals[cond_b]
        u_stat, p_raw = stats.mannwhitneyu(a, b, alternative="two-sided")
        p_bonf = min(float(p_raw) * n_comparisons, 1.0)
        med_diff = float(np.median(b)) - float(np.median(a))
        rows.append((
            CONDITION_LABELS[cond_a],
            CONDITION_LABELS[cond_b],
            u_stat,
            float(p_raw),
            med_diff,
            p_bonf,
        ))

    pair_width = max(len(f"{la} vs {lb}") for la, lb, *_ in rows)
    for la, lb, u, p_raw, med_diff, p_bonf in rows:
        pair_str = f"{la} vs {lb}"
        print(
            f"  {pair_str:<{pair_width}}  "
            f"U={u:5.1f}  p={p_raw:.4f}  "
            f"Δmedian={med_diff:+.4f}  p_bonf={p_bonf:.4f}"
        )

    # Resolution floor: smallest achievable two-sided p at n=5 vs n=5
    _, p_floor = stats.mannwhitneyu([0] * 5, [1] * 5, alternative="two-sided")
    print()
    print("-" * 65)
    print(
        f"Smallest achievable two-sided Mann-Whitney p at n=5 vs n=5: "
        f"{p_floor:.4f}  (U=0, complete separation)"
    )
    print()


# ─── CSV export ──────────────────────────────────────────────────────────────

def _write_final_fitness_csv(curves: dict[str, list[np.ndarray]]) -> None:
    import csv
    grouped = runs_by_condition()
    out = FIGS_DIR / "replication_final_fitness.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["condition", "label", "seed", "final_best_fitness"])
        for cond in CONDITION_ORDER:
            for run, series in zip(grouped[cond], curves[cond]):
                writer.writerow([cond, CONDITION_LABELS[cond], run["seed"], float(series[-1])])
    print(f"Saved: {out}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    curves, finals = _load_and_validate()
    _fitness_curves_figure(curves)
    _final_box_figure(finals)
    _statistical_summary(finals)
    _write_final_fitness_csv(curves)


if __name__ == "__main__":
    main()
