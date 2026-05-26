"""Per-neuron viable-range diagnostics across evolutionary generations.

For every completed run, replays best-per-generation individuals sampled every
10 generations (always including gen 0 and the final generation). Each replay
runs twice: once under the run's training HP-mode and once with HP disabled, so
both trained dynamics and raw genotype-only dynamics are captured.

At each timestep each neuron is classified as:
  U — under-active  (z < H_L = 0.2)
  V — viable        (H_L <= z <= H_U = 0.8)
  O — over-active   (z > H_U = 0.8)

Outputs:
  figs/viable_range_diagnostics.npz   – all metrics as labelled arrays
  figs/viable_range_states.pdf        – state-fraction evolution (5×4 grid)
  figs/viable_range_dwell.pdf         – viable-state dwell-time evolution
  figs/viable_range_transitions.pdf   – entry-exit rate evolution

Run from repo root:
    uv run python scripts/analysis/viable_range_diagnostics.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from agent.body import AgentBody
from ctrnn.agent import CTRNNAgent
from environment.trial import run_trial
from experiments.config import Condition, run_config_from_json  # noqa: F401
from plasticity.hp import HP

from load_runs import runs_by_condition
from plot_utils import (
    COLOUR_OVER_LINE, COLOUR_UNDER_LINE, COLOUR_VIABLE_LINE,
    CONDITION_LABELS, CONDITION_ORDER,
    FIRING_RATE_CMAP,  # noqa: F401 – exported for callers
    H_L, H_U,
    NEURON_LABELS,
)

_REPO_ROOT = Path(os.path.realpath(__file__)).parent.parent.parent
FIGS_DIR   = _REPO_ROOT / "figs"

# ── Shared constants ───────────────────────────────────────────────────────────

_CONDITION_TO_HP_MODE: dict[str, str] = {
    "HP_OFF":             "none",
    "HP_DEV_ONLY":        "development",
    "HP_BEHAVIOUR_ONLY":  "behaviour",
    "HP_BOTH":            "both",
}

SAMPLE_STEP   = 10
SHARED_SEED   = 42
N_SHAPES      = 20
N_NEURONS     = 5
MODE_TRAINING = 0   # index for the run's own training HP-mode
MODE_HP_OFF   = 1   # index for hp_mode='none'
N_MODES       = 2

# Line style for each HP mode (colour is always COLOUR_VIABLE_LINE; mode shown by dash)
_MODE_LS    = ["-",  "--"]
_MODE_LABEL = ["Training mode", "HP-off"]


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ── Generation sampling ────────────────────────────────────────────────────────

def _sampled_gen_items(run: dict) -> list[tuple[int, Path]]:
    """Return [(gen_idx, path)] sampled every SAMPLE_STEP, always including 0 and final."""
    best_dir  = run["run_dir"] / "best_per_gen"
    all_files = {int(p.stem[4:]): p for p in best_dir.glob("gen_*.npz")}
    if not all_files:
        raise FileNotFoundError(f"No best_per_gen files in {best_dir}")
    final_gen  = max(all_files)
    sampled    = set(range(0, final_gen + 1, SAMPLE_STEP)) | {0, final_gen}
    return sorted((g, all_files[g]) for g in sampled if g in all_files)


# ── Metric computation ─────────────────────────────────────────────────────────

def _mean_dwell(state_bool: np.ndarray) -> float:
    """Mean length of consecutive True runs in state_bool (1-D)."""
    if not state_bool.any():
        return 0.0
    padded = np.concatenate([[False], state_bool, [False]])
    diff   = np.diff(padded.astype(np.int8))
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]
    return float((ends - starts).mean())


def compute_metrics(neural: np.ndarray) -> dict:
    """
    Compute per-neuron three-state metrics from a (T, 5) firing-rate array.

    Returns a dict; all arrays are shape (N_NEURONS,) except 'trans' (N_NEURONS, 3, 3).
    State indices: 0 = U, 1 = V, 2 = O.
    """
    T = neural.shape[0]

    U  = neural < H_L               # (T, 5)
    O  = neural > H_U
    V  = ~U & ~O

    frac_U = U.mean(axis=0)         # (5,)
    frac_V = V.mean(axis=0)
    frac_O = O.mean(axis=0)

    # State index sequence: 0 = U, 1 = V, 2 = O
    seq = np.where(U, 0, np.where(V, 1, 2))   # (T, 5)

    dwell_U = np.array([_mean_dwell(U[:, n]) for n in range(N_NEURONS)])
    dwell_V = np.array([_mean_dwell(V[:, n]) for n in range(N_NEURONS)])
    dwell_O = np.array([_mean_dwell(O[:, n]) for n in range(N_NEURONS)])

    # Transition counts → rates per 1 000 timesteps
    from_s  = seq[:-1]              # (T-1, 5)
    to_s    = seq[1:]
    changed = from_s != to_s        # (T-1, 5) boolean

    trans = np.zeros((N_NEURONS, 3, 3), dtype=float)
    for n in range(N_NEURONS):
        mask = changed[:, n]
        if mask.any():
            np.add.at(trans[n], (from_s[mask, n], to_s[mask, n]), 1)
    trans *= 1000.0 / T             # convert counts to rate per 1 000 steps

    # Direct crossing: U→O or O→U, skipping viable range, per 1 000 steps
    direct_cross = trans[:, 0, 2] + trans[:, 2, 0]   # (5,)

    return {
        "frac_U": frac_U, "frac_V": frac_V, "frac_O": frac_O,
        "dwell_U": dwell_U, "dwell_V": dwell_V, "dwell_O": dwell_O,
        "trans": trans,
        "direct_cross": direct_cross,
    }


# ── Data collection ────────────────────────────────────────────────────────────

def collect_data(grouped: dict) -> dict:
    """
    Replay all runs across sampled generations and return metrics as numpy arrays.

    Array layout (axes in order):
        0  condition  (len CONDITION_ORDER = 4)
        1  run        (up to 5 per condition)
        2  gen        (n_sampled_gens, indexed by sampled_gen_indices)
        3  neuron     (N_NEURONS = 5)
        4  mode       (0 = training, 1 = HP-off)
    """
    # Reference sampled-gen indices from first available run (same for all)
    first_run   = next(r for c in CONDITION_ORDER for r in grouped[c])
    ref_items   = _sampled_gen_items(first_run)
    sgi         = np.array([g for g, _ in ref_items], dtype=int)
    n_gens      = len(sgi)
    g2i         = {int(g): i for i, g in enumerate(sgi)}

    n_cond  = len(CONDITION_ORDER)
    n_runs  = max(len(grouped[c]) for c in CONDITION_ORDER)
    base    = (n_cond, n_runs, n_gens, N_NEURONS, N_MODES)

    frac_U  = np.full(base, np.nan)
    frac_V  = np.full(base, np.nan)
    frac_O  = np.full(base, np.nan)
    dwell_U = np.full(base, np.nan)
    dwell_V = np.full(base, np.nan)
    dwell_O = np.full(base, np.nan)
    trans   = np.full(base + (3, 3), np.nan)
    direct  = np.full(base, np.nan)

    for ci, cond in enumerate(CONDITION_ORDER):
        print(f"[{_ts()}] Condition: {CONDITION_LABELS[cond]} ({cond})", flush=True)

        for ri, run in enumerate(grouped[cond]):
            items   = _sampled_gen_items(run)
            config  = run_config_from_json(run["run_dir"] / "config.json")
            t_mode  = _CONDITION_TO_HP_MODE[config.condition.value]

            print(
                f"  [{_ts()}] Run: {run['run_dir'].name}"
                f"  seed={run['seed']}  sampled_gens={len(items)}",
                flush=True,
            )

            for gen_idx, gen_file in items:
                ai = g2i.get(gen_idx)
                if ai is None:
                    continue

                with np.load(gen_file) as d:
                    genotype = d["genotype"].copy()

                for mi, hp_mode in enumerate([t_mode, "none"]):
                    agent  = CTRNNAgent(config.ctrnn)
                    agent.load_genotype(genotype)
                    hp     = HP(
                        H_L=config.hp.h_low, H_U=config.hp.h_high,
                        tau_w=config.hp.tau_w, tau_b=config.hp.tau_b,
                    )
                    rng    = np.random.default_rng(SHARED_SEED)
                    body   = AgentBody()
                    record = run_trial(
                        agent, hp, body, rng, n_shapes=N_SHAPES, hp_mode=hp_mode
                    )
                    neural = np.concatenate(record.neural_states, axis=0)
                    m      = compute_metrics(neural)

                    frac_U [ci, ri, ai, :, mi] = m["frac_U"]
                    frac_V [ci, ri, ai, :, mi] = m["frac_V"]
                    frac_O [ci, ri, ai, :, mi] = m["frac_O"]
                    dwell_U[ci, ri, ai, :, mi] = m["dwell_U"]
                    dwell_V[ci, ri, ai, :, mi] = m["dwell_V"]
                    dwell_O[ci, ri, ai, :, mi] = m["dwell_O"]
                    trans  [ci, ri, ai, :, mi] = m["trans"]
                    direct [ci, ri, ai, :, mi] = m["direct_cross"]

                print(
                    f"    gen {gen_idx:4d}  training={t_mode:<15}  hp_off=done",
                    flush=True,
                )

    return dict(
        sampled_gen_indices=sgi,
        condition_order=np.array(CONDITION_ORDER),
        frac_U=frac_U, frac_V=frac_V, frac_O=frac_O,
        dwell_U=dwell_U, dwell_V=dwell_V, dwell_O=dwell_O,
        transition_rates=trans,
        direct_crossing_rate=direct,
    )


# ── Derived metric ─────────────────────────────────────────────────────────────

def compute_entry_exit_rate(trans: np.ndarray) -> np.ndarray:
    """Mean of V→U, V→O, U→V, O→V rates per 1000 timesteps.

    trans shape: (..., 3, 3), state indices 0=U 1=V 2=O.
    Output shape: trans.shape[:-2].
    """
    return np.nanmean(
        np.stack([
            trans[..., 1, 0],   # V → U
            trans[..., 1, 2],   # V → O
            trans[..., 0, 1],   # U → V
            trans[..., 2, 1],   # O → V
        ], axis=-1),
        axis=-1,
    )


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _apply_panel_style(ax: plt.Axes, sgi: np.ndarray) -> None:
    ax.set_xlim(int(sgi[0]), int(sgi[-1]))
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_grid_labels(
    fig: plt.Figure,
    axes: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    row_ylabel: str,
) -> None:
    """Set column titles and row ylabel annotations on a 5×4 axes grid."""
    n_rows, n_cols = axes.shape
    for ci in range(n_cols):
        axes[0, ci].set_title(col_labels[ci], fontsize=9, fontweight="bold", pad=5)
    for ni in range(n_rows):
        axes[ni, 0].set_ylabel(
            f"{row_labels[ni]}\n{row_ylabel}",
            fontsize=7.5, fontweight="bold", labelpad=8,
        )
    for ni in range(n_rows - 1):
        for ci in range(n_cols):
            axes[ni, ci].tick_params(axis="x", labelbottom=False)
    for ci in range(n_cols):
        axes[-1, ci].set_xlabel("Generation", fontsize=8)


def _annotate_clipped(ax: plt.Axes, panel_max: float, cap: float, fmt: str = ".0f") -> None:
    """Add a small italic note if panel_max exceeds the row cap."""
    if panel_max > cap * 1.001:
        ax.text(
            0.97, 0.97, f"↑{panel_max:{fmt}}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5, style="italic", color="#555555",
        )


def _states_row_cap(fV: np.ndarray, fU: np.ndarray, fO: np.ndarray, ni: int) -> float:
    """95th percentile of raw state-fraction values for neuron row ni (all conds/modes)."""
    row = np.concatenate([
        fV[:, :, :, ni, :].ravel(),
        fU[:, :, :, ni, :].ravel(),
        fO[:, :, :, ni, :].ravel(),
    ])
    return float(np.nanpercentile(row, 95))


def _dwell_row_cap(dV: np.ndarray, ni: int) -> float:
    """95th percentile of plotted dwell values (median + q75) for neuron row ni."""
    vals: list[float] = []
    for ci in range(len(CONDITION_ORDER)):
        for mi in range(N_MODES):
            panel = dV[ci, :, :, ni, mi]
            vals.extend(np.nanmedian(panel, axis=0).tolist())
            vals.extend(np.nanpercentile(panel, 75, axis=0).tolist())
    return float(np.nanpercentile([v for v in vals if not np.isnan(v)], 95))


# ── Figure 1: state fractions ──────────────────────────────────────────────────

def build_states_figure(data: dict) -> plt.Figure:
    """5×4 grid: frac_V (both modes), frac_U and frac_O (training only), p95-capped rows."""
    sgi = data["sampled_gen_indices"]
    fU, fV, fO = data["frac_U"], data["frac_V"], data["frac_O"]

    # Per-row p95 cap from raw fraction data (naturally ≤ 1; prevents scale lock to 1.0)
    row_caps = [_states_row_cap(fV, fU, fO, ni) for ni in range(N_NEURONS)]

    fig, axes = plt.subplots(
        N_NEURONS, len(CONDITION_ORDER),
        figsize=(15, 13),
        sharex=True, sharey="row",
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for ni in range(N_NEURONS):
        cap = row_caps[ni]
        for ci, cond in enumerate(CONDITION_ORDER):
            ax = axes[ni, ci]

            # frac_V training: solid line + ±1 SD band
            fV_tr     = fV[ci, :, :, ni, MODE_TRAINING]
            mean_V_tr = np.nanmean(fV_tr, axis=0)
            std_V_tr  = np.nanstd (fV_tr, axis=0, ddof=1)
            ax.fill_between(sgi, mean_V_tr - std_V_tr, mean_V_tr + std_V_tr,
                            color=COLOUR_VIABLE_LINE, alpha=0.18, zorder=1)
            ax.plot(sgi, mean_V_tr, color=COLOUR_VIABLE_LINE, lw=1.5, ls="-", zorder=3)

            # frac_V HP-off: dashed, no band (the assimilation signal)
            mean_V_hp = np.nanmean(fV[ci, :, :, ni, MODE_HP_OFF], axis=0)
            ax.plot(sgi, mean_V_hp, color=COLOUR_VIABLE_LINE, lw=1.2, ls="--", zorder=3)

            # frac_U training only: solid (HP-off derivable, excluded to reduce clutter)
            mean_U_tr = np.nanmean(fU[ci, :, :, ni, MODE_TRAINING], axis=0)
            ax.plot(sgi, mean_U_tr, color=COLOUR_UNDER_LINE, lw=1.1, ls="-", zorder=2)

            # frac_O training only: solid
            mean_O_tr = np.nanmean(fO[ci, :, :, ni, MODE_TRAINING], axis=0)
            ax.plot(sgi, mean_O_tr, color=COLOUR_OVER_LINE, lw=1.1, ls="-", zorder=2)

            panel_max = float(max(
                np.nanmax(mean_V_tr + std_V_tr),
                np.nanmax(mean_V_hp),
                np.nanmax(mean_U_tr),
                np.nanmax(mean_O_tr),
            ))
            _annotate_clipped(ax, panel_max, cap, fmt=".2f")
            _apply_panel_style(ax, sgi)

        axes[ni, 0].set_ylim(0, cap)

    _add_grid_labels(
        fig, axes,
        row_labels=NEURON_LABELS,
        col_labels=[CONDITION_LABELS[c] for c in CONDITION_ORDER],
        row_ylabel="State fraction",
    )

    handles = [
        Line2D([0], [0], color=COLOUR_VIABLE_LINE, lw=1.5, ls="-",
               label="V viable — training"),
        Line2D([0], [0], color=COLOUR_VIABLE_LINE, lw=1.2, ls="--",
               label="V viable — HP-off"),
        Line2D([0], [0], color=COLOUR_UNDER_LINE,  lw=1.1, ls="-",
               label="U under-active — training"),
        Line2D([0], [0], color=COLOUR_OVER_LINE,   lw=1.1, ls="-",
               label="O over-active — training"),
        Line2D([0], [0], color=COLOUR_VIABLE_LINE, lw=8, alpha=0.22,
               label="V ±1 SD"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center", ncol=5, fontsize=7.5,
        frameon=False, bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ── Figure 2: viable-state dwell time ─────────────────────────────────────────

def build_dwell_figure(data: dict) -> plt.Figure:
    """5×4 grid: median dwell_V with IQR band; y-axis capped at p95 of plotted values."""
    sgi = data["sampled_gen_indices"]
    dV  = data["dwell_V"]

    # Per-row p95 cap based on plotted median + q75 (not raw data; avoids single-run spikes)
    row_caps = [_dwell_row_cap(dV, ni) for ni in range(N_NEURONS)]

    fig, axes = plt.subplots(
        N_NEURONS, len(CONDITION_ORDER),
        figsize=(15, 13),
        sharex=True, sharey="row",
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for ni in range(N_NEURONS):
        cap = row_caps[ni]
        for ci, cond in enumerate(CONDITION_ORDER):
            ax = axes[ni, ci]
            panel_max_vals: list[float] = []
            for mi in range(N_MODES):
                panel = dV[ci, :, :, ni, mi]            # (n_runs, n_gens)
                med = np.nanmedian(panel, axis=0)
                q25 = np.nanpercentile(panel, 25, axis=0)
                q75 = np.nanpercentile(panel, 75, axis=0)
                # Training band slightly more opaque than HP-off band
                alpha = 0.20 if mi == MODE_TRAINING else 0.10
                ax.fill_between(sgi, q25, q75,
                                color=COLOUR_VIABLE_LINE, alpha=alpha, zorder=1)
                ax.plot(sgi, med, color=COLOUR_VIABLE_LINE, lw=1.3, ls=_MODE_LS[mi],
                        zorder=2)
                panel_max_vals.extend([float(np.nanmax(q75)), float(np.nanmax(med))])

            _annotate_clipped(ax, max(panel_max_vals), cap)
            _apply_panel_style(ax, sgi)

        axes[ni, 0].set_ylim(0, cap)

    _add_grid_labels(
        fig, axes,
        row_labels=NEURON_LABELS,
        col_labels=[CONDITION_LABELS[c] for c in CONDITION_ORDER],
        row_ylabel="Dwell time (steps)",
    )

    handles = [
        Line2D([0], [0], color=COLOUR_VIABLE_LINE, lw=1.3, ls="-",
               label="Training mode"),
        Line2D([0], [0], color=COLOUR_VIABLE_LINE, lw=1.3, ls="--",
               label="HP-off"),
        Line2D([0], [0], color=COLOUR_VIABLE_LINE, lw=8, alpha=0.22,
               label="IQR across runs"),
        Line2D([0], [0], color="none", lw=0,
               label="↑N  values above row cap exist"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center", ncol=4, fontsize=7.5,
        frameon=False, bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ── Figure 3: entry-exit transition rate ──────────────────────────────────────

def build_transitions_figure(data: dict, entry_exit: np.ndarray) -> plt.Figure:
    """5×4 grid: entry-exit rate across generations (mean ±1 SD across runs)."""
    sgi = data["sampled_gen_indices"]

    fig, axes = plt.subplots(
        N_NEURONS, len(CONDITION_ORDER),
        figsize=(15, 13),
        sharex=True, sharey="row",
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for ni in range(N_NEURONS):
        for ci, cond in enumerate(CONDITION_ORDER):
            ax = axes[ni, ci]
            for mi in range(N_MODES):
                panel = entry_exit[ci, :, :, ni, mi]    # (n_runs, n_gens)
                mean  = np.nanmean(panel, axis=0)
                std   = np.nanstd (panel, axis=0, ddof=1)
                alpha = 0.20 if mi == MODE_TRAINING else 0.10
                ax.fill_between(sgi, mean - std, mean + std,
                                color=COLOUR_VIABLE_LINE, alpha=alpha, zorder=1)
                ax.plot(sgi, mean, color=COLOUR_VIABLE_LINE, lw=1.3,
                        ls=_MODE_LS[mi], zorder=2)

            ax.set_ylim(bottom=0)
            _apply_panel_style(ax, sgi)

    _add_grid_labels(
        fig, axes,
        row_labels=NEURON_LABELS,
        col_labels=[CONDITION_LABELS[c] for c in CONDITION_ORDER],
        row_ylabel="Rate (per 1000 steps)",
    )

    handles = [
        Line2D([0], [0], color=COLOUR_VIABLE_LINE, lw=1.3, ls="-",
               label="Training mode"),
        Line2D([0], [0], color=COLOUR_VIABLE_LINE, lw=1.3, ls="--",
               label="HP-off"),
        Line2D([0], [0], color=COLOUR_VIABLE_LINE, lw=8, alpha=0.22,
               label="±1 SD across runs"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center", ncol=3, fontsize=7.5,
        frameon=False, bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(data: dict, entry_exit: np.ndarray) -> None:
    fU   = data["frac_U"]
    fV   = data["frac_V"]
    fO   = data["frac_O"]
    dV   = data["dwell_V"]
    sgi  = data["sampled_gen_indices"]
    fi   = len(sgi) - 1   # index of final sampled generation

    print()
    print("=" * 95)
    print(f"Summary at final generation (gen {int(sgi[fi])}): mean across runs × neurons")
    print("=" * 95)
    hdr = (
        f"  {'Condition':<22} {'Mode':<15}"
        f" {'frac_U':>7} {'frac_V':>7} {'frac_O':>7}"
        f" {'dwell_V':>9} {'entry_exit':>11}"
    )
    print(hdr)
    print("-" * 95)

    for ci, cond in enumerate(CONDITION_ORDER):
        for mi, mode_name in [(MODE_TRAINING, "Training"), (MODE_HP_OFF, "HP off")]:
            fu = np.nanmean(fU [ci, :, fi, :, mi])
            fv = np.nanmean(fV [ci, :, fi, :, mi])
            fo = np.nanmean(fO [ci, :, fi, :, mi])
            dv = np.nanmedian(dV[ci, :, fi, :, mi])    # median — dwell is right-skewed
            ee = np.nanmean(entry_exit[ci, :, fi, :, mi])
            print(
                f"  {CONDITION_LABELS[cond]:<22} {mode_name:<15}"
                f" {fu:7.4f} {fv:7.4f} {fo:7.4f}"
                f" {dv:9.2f} {ee:11.4f}"
            )
        print()


# ── CSV export ────────────────────────────────────────────────────────────────

def _write_viable_range_csv(data: dict, entry_exit: np.ndarray) -> None:
    import csv
    fU  = data["frac_U"]
    fV  = data["frac_V"]
    fO  = data["frac_O"]
    dV  = data["dwell_V"]
    sgi = data["sampled_gen_indices"]
    mode_names = ["training", "HP-off"]

    out = FIGS_DIR / "viable_range_summary.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "condition", "hp_mode", "generation",
            "mean_frac_U", "mean_frac_V", "mean_frac_O",
            "median_dwell_V", "mean_entry_exit",
        ])
        for ci, cond in enumerate(CONDITION_ORDER):
            for mi, mode_name in enumerate(mode_names):
                for ai, gen in enumerate(sgi):
                    writer.writerow([
                        CONDITION_LABELS[cond],
                        mode_name,
                        int(gen),
                        float(np.nanmean(fU[ci, :, ai, :, mi])),
                        float(np.nanmean(fV[ci, :, ai, :, mi])),
                        float(np.nanmean(fO[ci, :, ai, :, mi])),
                        float(np.nanmedian(dV[ci, :, ai, :, mi])),
                        float(np.nanmean(entry_exit[ci, :, ai, :, mi])),
                    ])
    print(f"[{_ts()}] Saved: {out}", flush=True)


# ── Entry point ────────────────────────────────────────────────────────────────

def _load_or_collect(grouped: dict) -> dict:
    """Return data dict from npz if transition_rates present, else rerun replays."""
    npz_path = FIGS_DIR / "viable_range_diagnostics.npz"
    if npz_path.exists():
        d = np.load(npz_path, allow_pickle=False)
        if "transition_rates" in d:
            print(
                f"[{_ts()}] Loaded cached data from {npz_path} "
                f"(transition_rates present — skipping replays).",
                flush=True,
            )
            return {k: d[k] for k in d.files}

    print(f"[{_ts()}] transition_rates missing from npz; rerunning replay computation.", flush=True)
    data = collect_data(grouped)

    total = data["frac_U"] + data["frac_V"] + data["frac_O"]
    max_err = float(np.nanmax(np.abs(total - 1.0)))
    print(f"[{_ts()}] Fraction sum check: max |U+V+O − 1| = {max_err:.2e}", flush=True)
    assert max_err < 1e-10, f"Fractions do not sum to 1 (max error {max_err:.2e})"

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **data)
    print(f"[{_ts()}] Saved: {npz_path}", flush=True)
    return data


def main() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    grouped = runs_by_condition()

    data = _load_or_collect(grouped)

    # Derive entry-exit rate from the full transition-rate matrix
    entry_exit = compute_entry_exit_rate(data["transition_rates"])

    print_summary(data, entry_exit)

    fig1 = build_states_figure(data)
    out1 = FIGS_DIR / "viable_range_states.pdf"
    fig1.savefig(out1, bbox_inches="tight")
    plt.close(fig1)
    print(f"[{_ts()}] Saved: {out1}", flush=True)

    fig2 = build_dwell_figure(data)
    out2 = FIGS_DIR / "viable_range_dwell.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)
    print(f"[{_ts()}] Saved: {out2}", flush=True)

    fig3 = build_transitions_figure(data, entry_exit)
    out3 = FIGS_DIR / "viable_range_transitions.pdf"
    fig3.savefig(out3, bbox_inches="tight")
    plt.close(fig3)
    print(f"[{_ts()}] Saved: {out3}", flush=True)

    _write_viable_range_csv(data, entry_exit)


if __name__ == "__main__":
    main()
