"""Frozen-HP test: compare HP-active vs HP-frozen fitness for all 20 runs.

For each run, loads the final-generation best individual from best_per_gen/
and evaluates it under two conditions:

  HP-active:  evolved hp_mode (plasticity running as intended)
  HP-frozen:  adiabatic elimination — developmental HP retained but
              behaviour-phase plasticity disabled

A variance baseline of 20 HP-active evaluations (seeds 0–19) provides a
noise floor against which the fitness drop is expressed in SD units.

Condition-specific evaluation windows and modes
  HP_OFF            active=none,        frozen=none,        shapes 0–19
  HP_DEV_ONLY       active=development, frozen=none,        shapes 0–19
  HP_BEHAVIOUR_ONLY active=behaviour,   frozen=none,        shapes 10–19
  HP_BOTH           active=both,        frozen=development, shapes 0–19

Outputs:
  figs/frozen_hp_scatter.pdf
  figs/frozen_hp_drop.pdf
  figs/frozen_hp_results.csv

Run from repo root:
    uv run python scripts/analysis/frozen_hp_test.py
"""

import csv
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent.body import AgentBody
from ctrnn.agent import CTRNNAgent
from environment.trial import run_trial
from experiments.config import run_config_from_json
from plasticity.hp import HP

from load_runs import CONDITION_ORDER, discover_runs
from plot_utils import CONDITION_LABELS

_REPO_ROOT = Path(os.path.realpath(__file__)).parent.parent.parent
FIGS_DIR = _REPO_ROOT / "figs"

# ── Visual constants ──────────────────────────────────────────────────────────

_COND_COLOURS: dict[str, str] = dict(
    zip(CONDITION_ORDER, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
)

# ── Measurement parameters ────────────────────────────────────────────────────

FAIR_SEED = 42          # single seed used for the HP-frozen measurement
N_BASELINE = 20         # seeds 0..19 for the HP-active variance baseline

_ACTIVE_MODE: dict[str, str] = {
    "HP_OFF":            "none",
    "HP_DEV_ONLY":       "development",
    "HP_BEHAVIOUR_ONLY": "behaviour",
    "HP_BOTH":           "both",
}
_FROZEN_MODE: dict[str, str] = {
    "HP_OFF":            "none",
    "HP_DEV_ONLY":       "none",
    "HP_BEHAVIOUR_ONLY": "none",
    "HP_BOTH":           "development",
}
# Python slice objects for the shape evaluation window
_SHAPE_SLICE: dict[str, slice] = {
    "HP_OFF":            slice(0, 20),
    "HP_DEV_ONLY":       slice(0, 20),
    "HP_BEHAVIOUR_ONLY": slice(10, 20),
    "HP_BOTH":           slice(0, 20),
}


# ── Local fitness helpers (exact mirror of environment/fitness.py) ─────────────

def _phi(x: float) -> float:
    return x if 0.0 <= x <= 1.0 else 0.0


def _shape_contribution(shape_init, body_xs_i, shape_xs_i) -> float:
    shape_x0, vx, vy, agent_x0 = shape_init
    S0 = abs(agent_x0 - shape_x0)
    Sf = abs(body_xs_i[-1] - shape_xs_i[-1])
    Smax = (1.0 + abs(vx)) * (100.0 / abs(vy))

    first = 0.0 if S0 == 0.0 else _phi(1.0 - Sf / S0)
    second = max(0.0, min(1.0, 1.0 - Sf / Smax))
    return 0.5 * (first + second)


def _aggregate(record, shape_slice: slice) -> float:
    indices = range(*shape_slice.indices(len(record.shape_inits)))
    return float(np.mean([
        _shape_contribution(record.shape_inits[i], record.body_xs[i], record.shape_xs[i])
        for i in indices
    ]))


# ── Run-file helpers ──────────────────────────────────────────────────────────

def _final_gen_file(run_dir: Path) -> Path:
    best_dir = run_dir / "best_per_gen"
    files = sorted(best_dir.glob("gen_*.npz"), key=lambda p: int(p.stem[4:]))
    if not files:
        raise FileNotFoundError(f"No best_per_gen files in {best_dir}")
    return files[-1]


def _build_agent_hp(run_dir: Path, config) -> tuple:
    """Return (agent, hp, final_gen_fitness) for the best individual in run_dir."""
    npz = _final_gen_file(run_dir)
    with np.load(npz) as data:
        genotype = data["genotype"].copy()
        final_fitness = float(data["fitness"])

    agent = CTRNNAgent(config.ctrnn)
    agent.load_genotype(genotype)

    hp = HP(
        H_L=config.hp.h_low,
        H_U=config.hp.h_high,
        tau_w=config.hp.tau_w,
        tau_b=config.hp.tau_b,
    )
    return agent, hp, final_fitness


def _evaluate(agent, hp, hp_mode: str, seed: int, shape_slice: slice) -> float:
    """Single run_trial evaluation, aggregated over shape_slice."""
    rng = np.random.default_rng(seed=seed)
    body = AgentBody()
    record = run_trial(agent, hp, body, rng, n_shapes=20, hp_mode=hp_mode)
    return _aggregate(record, shape_slice)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = discover_runs()
    all_runs.sort(key=lambda r: (r["condition"], r["seed"]))
    print(f"[{_ts()}] Discovered {len(all_runs)} runs total.")

    rows: list[dict] = []

    for cond in CONDITION_ORDER:
        cond_runs = [r for r in all_runs if r["condition"] == cond]
        label = CONDITION_LABELS[cond]
        print(f"\n[{_ts()}] Condition: {label} ({cond})  —  {len(cond_runs)} runs")

        active_mode = _ACTIVE_MODE[cond]
        frozen_mode = _FROZEN_MODE[cond]
        shape_slice = _SHAPE_SLICE[cond]

        for run in cond_runs:
            run_dir = run["run_dir"]
            seed = run["seed"]
            print(f"  [{_ts()}] {run_dir.name}  seed={seed}")

            config = run_config_from_json(run_dir / "config.json")
            agent, hp, final_fitness = _build_agent_hp(run_dir, config)

            # Variance baseline: 20 HP-active evaluations, seeds 0..19
            baseline = [
                _evaluate(agent, hp, active_mode, seed=s, shape_slice=shape_slice)
                for s in range(N_BASELINE)
            ]
            active_mean = float(np.mean(baseline))
            active_sd   = float(np.std(baseline, ddof=1))

            # HP-frozen: single evaluation with the fair seed
            frozen_score = _evaluate(
                agent, hp, frozen_mode, seed=FAIR_SEED, shape_slice=shape_slice
            )

            drop = active_mean - frozen_score
            drop_sd = drop / active_sd if active_sd > 0.0 else float("nan")

            print(
                f"    active mean={active_mean:.4f}  SD={active_sd:.4f}"
                f"  frozen={frozen_score:.4f}"
                f"  drop={drop:+.4f}  ({drop_sd:+.2f} SD)"
            )

            rows.append({
                "condition":       cond,
                "run_dir":         str(run_dir),
                "seed":            seed,
                "final_gen_fitness": final_fitness,
                "active_mean":     active_mean,
                "active_sd":       active_sd,
                "frozen_score":    frozen_score,
                "drop":            drop,
                "drop_sd_units":   drop_sd,
            })

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = FIGS_DIR / "frozen_hp_results.csv"
    _FIELDS = [
        "condition", "run_dir", "seed", "final_gen_fitness",
        "active_mean", "active_sd", "frozen_score", "drop", "drop_sd_units",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[{_ts()}] Saved: {csv_path}")

    # ── Figure 1: Scatter (HP-active mean vs HP-frozen) ───────────────────────
    fig1, ax1 = plt.subplots(figsize=(6, 6))

    for cond in CONDITION_ORDER:
        cond_rows = [r for r in rows if r["condition"] == cond]
        if not cond_rows:
            continue
        xs    = [r["active_mean"]  for r in cond_rows]
        ys    = [r["frozen_score"] for r in cond_rows]
        xerrs = [r["active_sd"]    for r in cond_rows]
        ax1.errorbar(
            xs, ys, xerr=xerrs,
            fmt="o",
            color=_COND_COLOURS[cond],
            label=CONDITION_LABELS[cond],
            capsize=4,
            elinewidth=1.0,
            markersize=6,
            zorder=3,
        )

    all_vals = [r["active_mean"] for r in rows] + [r["frozen_score"] for r in rows]
    lo = min(all_vals) - 0.03
    hi = max(all_vals) + 0.03
    ax1.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, zorder=1)
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_xlabel("HP-active fitness")
    ax1.set_ylabel("HP-frozen fitness")
    ax1.legend(frameon=False, fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig1.tight_layout()
    scatter_path = FIGS_DIR / "frozen_hp_scatter.pdf"
    fig1.savefig(scatter_path, bbox_inches="tight")
    plt.close(fig1)
    print(f"[{_ts()}] Saved: {scatter_path}")

    # ── Figure 2: Box plot of fitness drops by condition ──────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    drop_by_cond = [
        [r["drop"] for r in rows if r["condition"] == cond]
        for cond in CONDITION_ORDER
    ]
    tick_labels = [CONDITION_LABELS[c] for c in CONDITION_ORDER]

    bp = ax2.boxplot(
        drop_by_cond,
        positions=range(len(CONDITION_ORDER)),
        patch_artist=True,
        widths=0.45,
        medianprops={"color": "black", "lw": 1.5},
        whiskerprops={"lw": 1.0},
        capprops={"lw": 1.0},
        flierprops={"marker": ""},
    )
    for patch, cond in zip(bp["boxes"], CONDITION_ORDER):
        patch.set_facecolor(_COND_COLOURS[cond])
        patch.set_alpha(0.55)

    rng_jitter = np.random.default_rng(0)
    for pos, (cond, drops) in enumerate(zip(CONDITION_ORDER, drop_by_cond)):
        jx = rng_jitter.uniform(-0.12, 0.12, size=len(drops))
        ax2.scatter(
            [pos + j for j in jx],
            drops,
            color=_COND_COLOURS[cond],
            s=30,
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
        )

    ax2.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax2.set_xticks(range(len(CONDITION_ORDER)))
    ax2.set_xticklabels(tick_labels)
    ax2.set_ylabel("Fitness drop (HP-active minus HP-frozen)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    box_path = FIGS_DIR / "frozen_hp_drop.pdf"
    fig2.savefig(box_path, bbox_inches="tight")
    plt.close(fig2)
    print(f"[{_ts()}] Saved: {box_path}")


if __name__ == "__main__":
    main()
