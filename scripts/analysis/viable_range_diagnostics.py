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
  figs/viable_range_states.pdf        – state-fraction evolution across gens
  figs/viable_range_dwell.pdf         – viable-state dwell-time evolution

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

from load_runs import CONDITION_LABELS, CONDITION_ORDER, runs_by_condition

_REPO_ROOT = Path(os.path.realpath(__file__)).parent.parent.parent
FIGS_DIR   = _REPO_ROOT / "figs"

# ── Shared constants ───────────────────────────────────────────────────────────

_CONDITION_TO_HP_MODE: dict[str, str] = {
    "HP_OFF":             "none",
    "HP_DEV_ONLY":        "development",
    "HP_BEHAVIOUR_ONLY":  "behaviour",
    "HP_BOTH":            "both",
}

H_L           = 0.2
H_U           = 0.8
SAMPLE_STEP   = 10
SHARED_SEED   = 42
N_SHAPES      = 20
N_NEURONS     = 5
MODE_TRAINING = 0   # index for the run's own training HP-mode
MODE_HP_OFF   = 1   # index for hp_mode='none'
N_MODES       = 2

_NEURON_LABELS  = [
    "Left sensor", "Centre sensor", "Right sensor", "Left motor", "Right motor",
]
_NEURON_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


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


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _panel_states(
    ax,
    fU_cr: np.ndarray,   # (n_runs, n_gens, N_NEURONS)
    fV_cr: np.ndarray,
    fO_cr: np.ndarray,
    sgi: np.ndarray,
    title: str,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    """Fill one panel of the state-fraction figure."""
    for n in range(N_NEURONS):
        col    = _NEURON_COLOURS[n]
        mean_V = np.nanmean(fV_cr[:, :, n], axis=0)         # (n_gens,)
        std_V  = np.nanstd (fV_cr[:, :, n], axis=0, ddof=1)
        mean_U = np.nanmean(fU_cr[:, :, n], axis=0)
        mean_O = np.nanmean(fO_cr[:, :, n], axis=0)

        ax.fill_between(sgi, mean_V - std_V, mean_V + std_V,
                        color=col, alpha=0.12)
        ax.plot(sgi, mean_V, color=col, lw=1.2,  ls="-",  label=_NEURON_LABELS[n])
        ax.plot(sgi, mean_U, color=col, lw=0.65, ls="--", alpha=0.55)
        ax.plot(sgi, mean_O, color=col, lw=0.65, ls=":",  alpha=0.55)

    ax.set_ylim(0, 1)
    ax.set_xlim(int(sgi[0]), int(sgi[-1]))
    ax.set_title(title, fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_ylabel:
        ax.set_ylabel("State fraction", fontsize=8)
    if show_xlabel:
        ax.set_xlabel("Generation", fontsize=8)


def _panel_dwell(
    ax,
    dV_cr: np.ndarray,   # (n_runs, n_gens, N_NEURONS)
    sgi: np.ndarray,
    title: str,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    """Fill one panel of the viable-state dwell-time figure."""
    # SD band: across-run variation of the mean-neuron dwell time
    per_run_mean = np.nanmean(dV_cr, axis=2)             # (n_runs, n_gens)
    band_mean    = np.nanmean(per_run_mean, axis=0)       # (n_gens,)
    band_std     = np.nanstd (per_run_mean, axis=0, ddof=1)
    ax.fill_between(sgi, band_mean - band_std, band_mean + band_std,
                    color="grey", alpha=0.18, zorder=1, label="Mean neuron ±1 SD")

    for n in range(N_NEURONS):
        mean_n = np.nanmean(dV_cr[:, :, n], axis=0)      # (n_gens,)
        ax.plot(sgi, mean_n, color=_NEURON_COLOURS[n], lw=1.2, zorder=2,
                label=_NEURON_LABELS[n])

    ax.set_xlim(int(sgi[0]), int(sgi[-1]))
    ax.set_title(title, fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_ylabel:
        ax.set_ylabel("Mean dwell time (steps)", fontsize=8)
    if show_xlabel:
        ax.set_xlabel("Generation", fontsize=8)


def _shared_states_legend(fig) -> None:
    neuron_handles = [
        Line2D([0], [0], color=_NEURON_COLOURS[n], lw=1.2, label=_NEURON_LABELS[n])
        for n in range(N_NEURONS)
    ]
    style_handles = [
        Line2D([0], [0], color="grey", lw=1.2,  ls="-",  label="V viable (mean ±1 SD)"),
        Line2D([0], [0], color="grey", lw=0.65, ls="--", alpha=0.7, label="U under-active"),
        Line2D([0], [0], color="grey", lw=0.65, ls=":",  alpha=0.7, label="O over-active"),
    ]
    fig.legend(
        handles=neuron_handles + style_handles,
        loc="lower center", ncol=8, fontsize=7.5,
        frameon=False, bbox_to_anchor=(0.5, 0.0),
    )


def _shared_dwell_legend(fig) -> None:
    handles = [
        Line2D([0], [0], color=_NEURON_COLOURS[n], lw=1.2, label=_NEURON_LABELS[n])
        for n in range(N_NEURONS)
    ]
    handles.append(
        Line2D([0], [0], color="grey", lw=8, alpha=0.25, label="Mean neuron ±1 SD")
    )
    fig.legend(
        handles=handles,
        loc="lower center", ncol=6, fontsize=7.5,
        frameon=False, bbox_to_anchor=(0.5, 0.0),
    )


def build_states_figure(data: dict) -> plt.Figure:
    """Figure 1: three-state fraction evolution across generations (2 rows × 4 cols)."""
    sgi = data["sampled_gen_indices"]
    fU, fV, fO = data["frac_U"], data["frac_V"], data["frac_O"]

    fig, axes = plt.subplots(
        2, 4, figsize=(16, 8),
        sharex=True, sharey=True,
    )
    fig.patch.set_facecolor("white")

    for mi, (mode_idx, row_label) in enumerate([
        (MODE_TRAINING, "Training mode"),
        (MODE_HP_OFF,   "HP off"),
    ]):
        for ci, cond in enumerate(CONDITION_ORDER):
            t_mode = _CONDITION_TO_HP_MODE[cond]
            if mode_idx == MODE_TRAINING:
                mode_str = f"hp_mode='{t_mode}'"
            else:
                mode_str = "hp_mode='none'"
            title = f"{CONDITION_LABELS[cond]}\n[{mode_str}]"

            _panel_states(
                axes[mi, ci],
                fU[ci, :, :, :, mode_idx],
                fV[ci, :, :, :, mode_idx],
                fO[ci, :, :, :, mode_idx],
                sgi, title,
                show_ylabel=(ci == 0),
                show_xlabel=(mi == 1),
            )

    _shared_states_legend(fig)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


def build_dwell_figure(data: dict) -> plt.Figure:
    """Figure 2: viable-state dwell-time evolution (2 rows × 4 cols)."""
    sgi = data["sampled_gen_indices"]
    dV  = data["dwell_V"]

    fig, axes = plt.subplots(
        2, 4, figsize=(16, 8),
        sharex=True,
    )
    fig.patch.set_facecolor("white")

    for mi, (mode_idx, row_label) in enumerate([
        (MODE_TRAINING, "Training mode"),
        (MODE_HP_OFF,   "HP off"),
    ]):
        for ci, cond in enumerate(CONDITION_ORDER):
            t_mode = _CONDITION_TO_HP_MODE[cond]
            if mode_idx == MODE_TRAINING:
                mode_str = f"hp_mode='{t_mode}'"
            else:
                mode_str = "hp_mode='none'"
            title = f"{CONDITION_LABELS[cond]}\n[{mode_str}]"

            _panel_dwell(
                axes[mi, ci],
                dV[ci, :, :, :, mode_idx],
                sgi, title,
                show_ylabel=(ci == 0),
                show_xlabel=(mi == 1),
            )

    _shared_dwell_legend(fig)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(data: dict) -> None:
    fU   = data["frac_U"]
    fV   = data["frac_V"]
    fO   = data["frac_O"]
    dV   = data["dwell_V"]
    dc   = data["direct_crossing_rate"]
    sgi  = data["sampled_gen_indices"]
    fi   = len(sgi) - 1   # index of final sampled generation

    print()
    print("=" * 90)
    print(f"Summary at final generation (gen {int(sgi[fi])}): mean across runs × neurons")
    print("=" * 90)
    hdr = (
        f"  {'Condition':<22} {'Mode':<15}"
        f" {'frac_U':>7} {'frac_V':>7} {'frac_O':>7}"
        f" {'dwell_V':>8} {'direct_x':>9}"
    )
    print(hdr)
    print("-" * 90)

    for ci, cond in enumerate(CONDITION_ORDER):
        for mi, mode_name in [(MODE_TRAINING, "Training"), (MODE_HP_OFF, "HP off")]:
            fu = np.nanmean(fU [ci, :, fi, :, mi])
            fv = np.nanmean(fV [ci, :, fi, :, mi])
            fo = np.nanmean(fO [ci, :, fi, :, mi])
            dv = np.nanmean(dV [ci, :, fi, :, mi])
            dx = np.nanmean(dc [ci, :, fi, :, mi])
            print(
                f"  {CONDITION_LABELS[cond]:<22} {mode_name:<15}"
                f" {fu:7.4f} {fv:7.4f} {fo:7.4f}"
                f" {dv:8.2f} {dx:9.4f}"
            )
        print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    grouped   = runs_by_condition()
    n_runs    = sum(len(grouped[c]) for c in CONDITION_ORDER)
    print(
        f"[{_ts()}] Starting viable-range diagnostics on"
        f" {n_runs} runs across {len(CONDITION_ORDER)} conditions.",
        flush=True,
    )

    data = collect_data(grouped)

    # ── Sanity check: fractions must sum to 1 ────────────────────────────────
    total = data["frac_U"] + data["frac_V"] + data["frac_O"]
    max_err = float(np.nanmax(np.abs(total - 1.0)))
    print(f"\n[{_ts()}] Fraction sum check: max |U+V+O − 1| = {max_err:.2e}", flush=True)
    assert max_err < 1e-10, f"Fractions do not sum to 1 (max error {max_err:.2e})"

    # ── Save npz ──────────────────────────────────────────────────────────────
    npz_path = FIGS_DIR / "viable_range_diagnostics.npz"
    np.savez(npz_path, **data)
    print(f"[{_ts()}] Saved: {npz_path}", flush=True)

    # ── Figures ───────────────────────────────────────────────────────────────
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

    print_summary(data)


if __name__ == "__main__":
    main()
