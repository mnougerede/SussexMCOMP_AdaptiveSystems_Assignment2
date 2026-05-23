"""Read-only data loader for completed evolutionary runs.

Discovers runs by scanning manifests under experiments/, groups them by
experimental condition, and provides per-run accessors for the saved
per-generation fitness data.

Run from repo root:
    uv run python scripts/analysis/load_runs.py
"""

import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import experiment_status as _exp_status_mod
from experiment_status import _resolve_output_dir

# Use realpath so that parent navigation works correctly regardless of whether
# this file was found via a non-normalised sys.path entry like "scripts/analysis/..".
_REPO_ROOT = Path(os.path.realpath(__file__)).parent.parent.parent
_EXPERIMENTS_DIR = _REPO_ROOT / "experiments"

# experiment_status.EXPERIMENTS_DIR may resolve incorrectly when imported via a
# non-normalised sys.path entry; fix it so the rglob fallback in _resolve_output_dir
# searches the right place (needed when stored output_dir paths are from another machine).
_exp_status_mod.EXPERIMENTS_DIR = _EXPERIMENTS_DIR

CONDITION_LABELS: dict[str, str] = {
    "HP_OFF": "No HP",
    "HP_DEV_ONLY": "Dev only",
    "HP_BEHAVIOUR_ONLY": "Behaviour only",
    "HP_BOTH": "Both",
}

CONDITION_ORDER: list[str] = ["HP_OFF", "HP_DEV_ONLY", "HP_BEHAVIOUR_ONLY", "HP_BOTH"]


def discover_runs(experiments_dir: Path | str | None = None) -> list[dict]:
    """Return completed runs as a list of records.

    Each record contains:
      run_dir   – Path to the run output directory
      condition – condition string from config.json (authoritative)
      label     – human-readable label from CONDITION_LABELS
      seed      – integer seed from config.json

    If a manifest entry's condition disagrees with the run's config.json a
    UserWarning is raised naming the run directory and both values; config.json
    wins.
    """
    scan_dir = Path(experiments_dir) if experiments_dir is not None else _EXPERIMENTS_DIR

    runs: list[dict] = []
    for manifest_path in sorted(scan_dir.rglob("manifest.json")):
        with open(manifest_path) as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError:
                continue
        for entry in entries:
            if entry.get("status") != "complete":
                continue
            run_dir = _resolve_output_dir(entry)
            if not run_dir.exists():
                continue
            config_path = run_dir / "config.json"
            if not config_path.exists():
                continue
            with open(config_path) as f:
                config = json.load(f)

            config_condition: str = config.get("condition", "")
            manifest_condition: str = entry.get("condition", "")
            if config_condition != manifest_condition:
                warnings.warn(
                    f"Condition mismatch for {run_dir.name!r}: "
                    f"manifest has {manifest_condition!r}, "
                    f"config.json has {config_condition!r}; "
                    f"using config.json value.",
                    UserWarning,
                    stacklevel=2,
                )

            condition = config_condition
            seed = int(config["seed"])
            runs.append({
                "run_dir": run_dir,
                "condition": condition,
                "label": CONDITION_LABELS.get(condition, condition),
                "seed": seed,
            })

    return runs


def runs_by_condition(experiments_dir: Path | str | None = None) -> dict[str, list[dict]]:
    """Return completed runs grouped by condition in CONDITION_ORDER.

    All four standard conditions are present as keys even when empty.
    Conditions with unexpected run counts are included without error.
    """
    grouped: dict[str, list[dict]] = {cond: [] for cond in CONDITION_ORDER}
    for run in discover_runs(experiments_dir):
        cond = run["condition"]
        if cond not in grouped:
            grouped[cond] = []
        grouped[cond].append(run)
    return grouped


def _sorted_history_files(run: dict) -> list[Path]:
    history_dir = run["run_dir"] / "history"
    return sorted(
        history_dir.glob("gen_*.npz"),
        key=lambda p: int(p.stem[4:]),
    )


def best_fitness_series(run: dict) -> np.ndarray:
    """Return best_fitness per generation as a 1-D array, in generation order."""
    values: list[float] = []
    for f in _sorted_history_files(run):
        with np.load(f) as data:
            values.append(float(data["best_fitness"]))
    return np.array(values)


def mean_fitness_series(run: dict) -> np.ndarray:
    """Return mean_fitness per generation as a 1-D array, in generation order."""
    values: list[float] = []
    for f in _sorted_history_files(run):
        with np.load(f) as data:
            values.append(float(data["mean_fitness"]))
    return np.array(values)


def population_fitnesses(run: dict) -> np.ndarray:
    """Return stacked population fitnesses, shape (n_generations, population_size)."""
    rows: list[np.ndarray] = []
    for f in _sorted_history_files(run):
        with np.load(f) as data:
            rows.append(data["fitnesses"].copy())
    return np.stack(rows)


def print_runs(experiments_dir: Path | str | None = None) -> None:
    """Print a verification table of all discovered runs."""
    runs = discover_runs(experiments_dir)
    runs_sorted = sorted(runs, key=lambda r: (r["condition"], r["seed"]))

    dir_width = max((len(str(r["run_dir"])) for r in runs_sorted), default=20)
    dir_width = max(dir_width, len("Run directory"))
    header = f"{'Run directory':<{dir_width}}  {'Condition':<25}  {'Seed':>6}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for run in runs_sorted:
        print(f"{str(run['run_dir']):<{dir_width}}  {run['condition']:<25}  {run['seed']:>6}")
    print(f"\nTotal: {len(runs)} runs discovered")


if __name__ == "__main__":
    print_runs()
