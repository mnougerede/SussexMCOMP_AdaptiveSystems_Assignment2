import json
from pathlib import Path

import numpy as np
import pytest

import experiments.evolve as evolve_module
from experiments.config import Condition, GAConfig, RunConfig
from experiments.evolve import run_experiment
from experiments.io import load_history


def _minimal_config(output_dir: str, n_gens: int, seed: int = 42) -> RunConfig:
    return RunConfig(
        ga=GAConfig(pop_size=8, n_gens=n_gens, n_runs=1),
        condition=Condition.HP_OFF,
        seed=seed,
        output_dir=output_dir,
    )


# ─── 1. Resumption correctness ────────────────────────────────────────────────

def test_resumption_matches_uninterrupted(tmp_path):
    N = 5
    manifest = tmp_path / "manifest.json"

    partial_dir = str(tmp_path / "run_partial")
    uninterrupted_dir = str(tmp_path / "run_uninterrupted")

    # First half: N generations
    run_experiment(_minimal_config(partial_dir, n_gens=N, seed=7), manifest)

    # Resume: extend to 2N generations (same output_dir, same seed)
    run_experiment(_minimal_config(partial_dir, n_gens=2 * N, seed=7), manifest)

    # Reference: single uninterrupted run of 2N generations with the same seed
    run_experiment(_minimal_config(uninterrupted_dir, n_gens=2 * N, seed=7), manifest)

    partial_history = load_history(partial_dir)
    uninterrupted_history = load_history(uninterrupted_dir)

    assert partial_history["fitnesses"].shape == (2 * N, 8)
    assert uninterrupted_history["fitnesses"].shape == (2 * N, 8)

    np.testing.assert_array_equal(
        partial_history["fitnesses"],
        uninterrupted_history["fitnesses"],
    )
    np.testing.assert_array_equal(
        partial_history["best_fitness"],
        uninterrupted_history["best_fitness"],
    )


# ─── 2. Git commit mismatch ───────────────────────────────────────────────────

def test_git_commit_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(evolve_module, "get_git_commit", lambda: "aaaa1234")

    config = _minimal_config(str(tmp_path / "run_gc"), n_gens=3)
    manifest = tmp_path / "manifest.json"
    run_experiment(config, manifest)

    # Simulate checking out a different commit
    monkeypatch.setattr(evolve_module, "get_git_commit", lambda: "bbbb5678")

    with pytest.raises(RuntimeError, match="aaaa1234"):
        run_experiment(config, manifest)

    with pytest.raises(RuntimeError, match="bbbb5678"):
        run_experiment(config, manifest)


def test_git_commit_mismatch_skipped_when_either_empty(tmp_path, monkeypatch):
    # If either hash is empty the guard must not fire
    monkeypatch.setattr(evolve_module, "get_git_commit", lambda: "aaaa1234")
    config = _minimal_config(str(tmp_path / "run_empty"), n_gens=2)
    manifest = tmp_path / "manifest.json"
    run_experiment(config, manifest)

    # Tamper config.json so saved commit is empty
    config_path = Path(config.output_dir) / "config.json"
    with open(config_path) as f:
        d = json.load(f)
    d["git_commit"] = ""
    with open(config_path, "w") as f:
        json.dump(d, f)

    # Should not raise even though current commit is non-empty
    run_experiment(_minimal_config(config.output_dir, n_gens=4), manifest)
