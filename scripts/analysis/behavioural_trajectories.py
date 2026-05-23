"""Phase 8c behavioural trajectory analysis.

For each condition, selects the run whose final-generation best fitness is
highest (across the 5 runs for that condition) as the representative
individual. Runs 3 trials for each representative using a shared RNG seed so
all four individuals experience the same sequence of falling shapes, making
behavioural comparisons direct.

Outputs:
  figs/behavioural_trajectories.pdf
  figs/behavioural_trajectories_summary.txt

Run from repo root:
    uv run python scripts/analysis/behavioural_trajectories.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from agent.body import AgentBody
from ctrnn.agent import CTRNNAgent
from environment.trial import run_trial
from experiments.config import Condition, run_config_from_json  # noqa: F401 – Condition imported per spec
from plasticity.hp import HP

from load_runs import CONDITION_LABELS, CONDITION_ORDER, runs_by_condition

_REPO_ROOT = Path(os.path.realpath(__file__)).parent.parent.parent
FIGS_DIR = _REPO_ROOT / "figs"

# ── Shared constants ──────────────────────────────────────────────────────────

_CONDITION_TO_HP_MODE: dict[str, str] = {
    "HP_OFF":             "none",
    "HP_DEV_ONLY":        "development",
    "HP_BEHAVIOUR_ONLY":  "behaviour",
    "HP_BOTH":            "both",
}

N_TRIALS    = 3
SHARED_SEED = 42
N_SHAPES    = 20
H_L         = 0.2
H_U         = 0.8

_COND_COLOURS   = dict(zip(CONDITION_ORDER, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))
_NEURON_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
_NEURON_LABELS  = [f"n{i}" for i in range(5)]


# ── Helper: locate best_per_gen file ─────────────────────────────────────────

def _final_gen_file(run: dict) -> Path:
    """Return the numerically highest gen_NNNN.npz in best_per_gen/."""
    best_dir = run["run_dir"] / "best_per_gen"
    files = sorted(best_dir.glob("gen_*.npz"), key=lambda p: int(p.stem[4:]))
    if not files:
        raise FileNotFoundError(f"No best_per_gen files in {best_dir}")
    return files[-1]


def _final_gen_fitness(run: dict) -> float:
    with np.load(_final_gen_file(run)) as data:
        return float(data["fitness"])


# ── Selection ─────────────────────────────────────────────────────────────────

def select_representatives(grouped: dict) -> dict:
    """Per condition: return the run with the highest final-generation fitness."""
    reps = {}
    for cond in CONDITION_ORDER:
        runs = grouped[cond]
        if not runs:
            raise RuntimeError(f"No complete runs for condition {cond}")
        reps[cond] = max(runs, key=_final_gen_fitness)
    return reps


# ── Agent construction ────────────────────────────────────────────────────────

def build_agent_and_hp(run: dict) -> tuple:
    """Return (agent, hp, hp_mode, fitness) ready for run_trial."""
    config = run_config_from_json(run["run_dir"] / "config.json")

    npz = _final_gen_file(run)
    with np.load(npz) as data:
        genotype = data["genotype"].copy()
        fitness  = float(data["fitness"])

    agent = CTRNNAgent(config.ctrnn)
    agent.load_genotype(genotype)

    hp = HP(
        H_L=config.hp.h_low,
        H_U=config.hp.h_high,
        tau_w=config.hp.tau_w,
        tau_b=config.hp.tau_b,
    )
    hp_mode = _CONDITION_TO_HP_MODE[config.condition.value]
    return agent, hp, hp_mode, fitness


# ── Trial execution ───────────────────────────────────────────────────────────

def run_trials(agent, hp, hp_mode: str) -> list:
    """Run N_TRIALS trials from a fresh shared-seed RNG. Returns list of TrialRecord."""
    rng = np.random.default_rng(SHARED_SEED)
    records = []
    for _ in range(N_TRIALS):
        body = AgentBody()
        records.append(
            run_trial(agent, hp, body, rng, n_shapes=N_SHAPES, hp_mode=hp_mode)
        )
    return records


# ── Plotting ──────────────────────────────────────────────────────────────────

def _fill_cell(
    ax_traj,
    ax_neural,
    record,
    colour: str,
    show_traj_legend: bool,
    show_neural_legend: bool,
    show_xlabel: bool,
    show_ylabels: bool,
) -> None:
    """Populate one cell's two-panel stack from a TrialRecord."""
    agent_x = np.concatenate(record.body_xs)
    shape_x = np.concatenate(record.shape_xs)
    neural  = np.concatenate(record.neural_states, axis=0)   # (T_total, 5)
    t = np.arange(len(agent_x))

    # Upper: x-position traces
    ax_traj.plot(t, agent_x, color=colour,    lw=0.8,  label="Agent x")
    ax_traj.plot(t, shape_x, color="#888888", lw=0.8,  ls="--", alpha=0.8, label="Shape x")
    ax_traj.tick_params(labelsize=6)
    ax_traj.tick_params(axis="x", labelbottom=False)
    ax_traj.spines["top"].set_visible(False)
    ax_traj.spines["right"].set_visible(False)
    if show_ylabels:
        ax_traj.set_ylabel("x pos", fontsize=7)
    if show_traj_legend:
        ax_traj.legend(fontsize=6, frameon=False, loc="upper right",
                       handlelength=1.2, borderpad=0.3, labelspacing=0.2)

    # Lower: per-neuron firing rates
    for i in range(5):
        ax_neural.plot(
            t, neural[:, i],
            color=_NEURON_COLOURS[i], lw=0.55, alpha=0.85,
            label=_NEURON_LABELS[i],
        )
    ax_neural.axhline(H_L, color="black", lw=0.7, ls="--", alpha=0.45)
    ax_neural.axhline(H_U, color="black", lw=0.7, ls="--", alpha=0.45)
    ax_neural.set_ylim(0, 1)
    ax_neural.tick_params(labelsize=6)
    ax_neural.spines["top"].set_visible(False)
    ax_neural.spines["right"].set_visible(False)
    if show_xlabel:
        ax_neural.set_xlabel("Time (steps)", fontsize=7)
    if show_ylabels:
        ax_neural.set_ylabel("Output", fontsize=7)
    if show_neural_legend:
        ax_neural.legend(
            fontsize=5.5, frameon=False, loc="upper right",
            handlelength=1.0, borderpad=0.2, labelspacing=0.15, ncol=3,
        )


def build_figure(cond_records: dict) -> plt.Figure:
    """Build and return the 4×3 trajectory figure."""
    n_rows = len(CONDITION_ORDER)
    n_cols = N_TRIALS

    fig = plt.figure(figsize=(12, 15))
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.65, wspace=0.38,
        left=0.15, right=0.97, top=0.95, bottom=0.04,
    )

    # Accumulate leftmost-column axis pairs for row label placement
    left_ax_top: list = []
    left_ax_bot: list = []

    for row_idx, cond in enumerate(CONDITION_ORDER):
        colour   = _COND_COLOURS[cond]
        is_last_row = (row_idx == n_rows - 1)

        for col_idx, record in enumerate(cond_records[cond]):
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1,
                subplot_spec=outer[row_idx, col_idx],
                hspace=0.07,
                height_ratios=[2, 1],
            )
            ax_traj   = fig.add_subplot(inner[0])
            ax_neural = fig.add_subplot(inner[1], sharex=ax_traj)

            is_left = (col_idx == 0)

            _fill_cell(
                ax_traj, ax_neural, record, colour,
                show_traj_legend=(row_idx == 0 and col_idx == 0),
                show_neural_legend=(row_idx == 0 and col_idx == 0),
                show_xlabel=is_last_row,
                show_ylabels=is_left,
            )

            if row_idx == 0:
                ax_traj.set_title(
                    f"Trial {col_idx + 1}",
                    fontsize=9, fontweight="bold", pad=5,
                )

            if is_left:
                left_ax_top.append(ax_traj)
                left_ax_bot.append(ax_neural)

    # Row labels — centred over the full outer row cell (top of ax_traj to bottom of ax_neural)
    for ax_t, ax_n, cond in zip(left_ax_top, left_ax_bot, CONDITION_ORDER):
        bbox_t = ax_t.get_position()
        bbox_n = ax_n.get_position()
        y_mid  = (bbox_t.y1 + bbox_n.y0) / 2
        fig.text(
            0.02, y_mid,
            CONDITION_LABELS[cond],
            va="center", ha="center",
            fontsize=9, fontweight="bold",
            rotation=90,
        )

    return fig


# ── Summary text ──────────────────────────────────────────────────────────────

def write_summary(path: Path, cond_records: dict, cond_info: dict) -> None:
    with open(path, "w") as f:
        f.write("Behavioural Trajectory Analysis — Reproducibility Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Shared RNG seed : {SHARED_SEED}\n")
        f.write(f"Trials per agent: {N_TRIALS}\n")
        f.write(f"Shapes per trial: {N_SHAPES}\n\n")

        for cond in CONDITION_ORDER:
            info = cond_info[cond]
            f.write(f"Condition : {CONDITION_LABELS[cond]} ({cond})\n")
            f.write(f"  Run directory    : {info['run_dir']}\n")
            f.write(f"  Seed             : {info['seed']}\n")
            f.write(f"  Final-gen fitness: {info['fitness']:.6f}\n")
            for t_idx, record in enumerate(cond_records[cond]):
                f.write(f"  Trial {t_idx + 1} shape_inits (shape_x0, vx, vy, agent_x0):\n")
                for s_idx, si in enumerate(record.shape_inits):
                    f.write(
                        f"    {s_idx:2d}: x0={si[0]:9.4f}  vx={si[1]:7.4f}"
                        f"  vy={si[2]:7.4f}  agent_x0={si[3]:9.4f}\n"
                    )
            f.write("\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    grouped = runs_by_condition()
    reps     = select_representatives(grouped)

    print(f"Found {len(reps)} representative individuals (one per condition):")
    cond_records: dict = {}
    cond_info:    dict = {}

    for cond in CONDITION_ORDER:
        run = reps[cond]
        agent, hp, hp_mode, fitness = build_agent_and_hp(run)
        print(
            f"  {CONDITION_LABELS[cond]:<20}  run={run['run_dir'].name}"
            f"  seed={run['seed']}  fitness={fitness:.4f}  hp_mode={hp_mode!r}"
        )
        records = run_trials(agent, hp, hp_mode)
        cond_records[cond] = records
        cond_info[cond] = {
            "run_dir": run["run_dir"],
            "seed":    run["seed"],
            "fitness": fitness,
        }

    total_trials = sum(len(r) for r in cond_records.values())
    print(f"\nRan {total_trials} trials total ({len(reps)} individuals × {N_TRIALS} trials).")

    fig = build_figure(cond_records)
    out_fig = FIGS_DIR / "behavioural_trajectories.pdf"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig}")

    out_txt = FIGS_DIR / "behavioural_trajectories_summary.txt"
    write_summary(out_txt, cond_records, cond_info)
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
