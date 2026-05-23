import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis"))
import load_runs

# Tiny scale used throughout: 3 generations, population of 4.
# Seeds are deliberately different from the real experiment seeds.
N_GENS = 3
POP_SIZE = 4


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_run(base: Path, run_id: str, condition: str, seed: int) -> Path:
    """Create a minimal run directory: config.json + history/ npz files."""
    run_dir = base / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"condition": condition, "seed": seed}))
    history = run_dir / "history"
    history.mkdir()
    rng = np.random.default_rng(seed + 1000)
    for gen in range(N_GENS):
        fitnesses = rng.random(POP_SIZE)
        np.savez(
            history / f"gen_{gen:04d}.npz",
            fitnesses=fitnesses,
            best_fitness=np.float64(fitnesses.max()),
            mean_fitness=np.float64(fitnesses.mean()),
        )
    return run_dir


def _manifest_entry(run_id: str, condition: str, seed: int, run_dir: Path) -> dict:
    return {
        "run_id": run_id,
        "condition": condition,
        "seed": seed,
        "status": "complete",
        "output_dir": str(run_dir),
    }


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def four_run_experiments(tmp_path):
    """One completed run per condition, using all four real condition strings."""
    batch = tmp_path / "experiments" / "batch_a"
    batch.mkdir(parents=True)
    specs = [
        ("run_off",  "HP_OFF",             11),
        ("run_dev",  "HP_DEV_ONLY",         22),
        ("run_beh",  "HP_BEHAVIOUR_ONLY",   33),
        ("run_both", "HP_BOTH",             44),
    ]
    entries = []
    for run_id, cond, seed in specs:
        rd = _make_run(batch, run_id, cond, seed)
        entries.append(_manifest_entry(run_id, cond, seed, rd))
    (batch / "manifest.json").write_text(json.dumps(entries))
    return tmp_path / "experiments"


# ─── 1. Discovery and grouping ────────────────────────────────────────────────

def test_all_synthetic_runs_discovered(four_run_experiments):
    runs = load_runs.discover_runs(four_run_experiments)
    assert len(runs) == 4


def test_runs_grouped_under_correct_conditions(four_run_experiments):
    grouped = load_runs.runs_by_condition(four_run_experiments)
    for cond in load_runs.CONDITION_ORDER:
        assert cond in grouped
        assert len(grouped[cond]) == 1
        assert grouped[cond][0]["condition"] == cond


def test_grouped_keys_in_condition_order(four_run_experiments):
    grouped = load_runs.runs_by_condition(four_run_experiments)
    assert list(grouped.keys())[:4] == load_runs.CONDITION_ORDER


# ─── 2. Condition and seed read from config.json ──────────────────────────────

def test_condition_read_from_config(four_run_experiments):
    runs = load_runs.discover_runs(four_run_experiments)
    by_seed = {r["seed"]: r for r in runs}
    assert by_seed[11]["condition"] == "HP_OFF"
    assert by_seed[22]["condition"] == "HP_DEV_ONLY"
    assert by_seed[33]["condition"] == "HP_BEHAVIOUR_ONLY"
    assert by_seed[44]["condition"] == "HP_BOTH"


def test_seed_read_from_config(four_run_experiments):
    runs = load_runs.discover_runs(four_run_experiments)
    seeds = {r["seed"] for r in runs}
    assert seeds == {11, 22, 33, 44}


def test_label_derived_from_condition(four_run_experiments):
    runs = load_runs.discover_runs(four_run_experiments)
    for run in runs:
        assert run["label"] == load_runs.CONDITION_LABELS[run["condition"]]


# ─── 3. Condition mismatch warning ────────────────────────────────────────────

def test_mismatch_raises_user_warning(tmp_path):
    batch = tmp_path / "experiments" / "batch_b"
    batch.mkdir(parents=True)
    # config.json says HP_DEV_ONLY; manifest says HP_OFF
    rd = _make_run(batch, "run_mismatch", "HP_DEV_ONLY", 99)
    entry = _manifest_entry("run_mismatch", "HP_OFF", 99, rd)
    (batch / "manifest.json").write_text(json.dumps([entry]))

    with pytest.warns(UserWarning, match="run_mismatch"):
        runs = load_runs.discover_runs(tmp_path / "experiments")

    assert len(runs) == 1


def test_mismatch_warning_names_both_conditions(tmp_path):
    batch = tmp_path / "experiments" / "batch_b"
    batch.mkdir(parents=True)
    rd = _make_run(batch, "run_mismatch", "HP_DEV_ONLY", 99)
    entry = _manifest_entry("run_mismatch", "HP_OFF", 99, rd)
    (batch / "manifest.json").write_text(json.dumps([entry]))

    with pytest.warns(UserWarning, match="HP_OFF") as rec:
        load_runs.discover_runs(tmp_path / "experiments")

    msg = str(rec[0].message)
    assert "HP_OFF" in msg
    assert "HP_DEV_ONLY" in msg


def test_mismatch_uses_config_value(tmp_path):
    batch = tmp_path / "experiments" / "batch_b"
    batch.mkdir(parents=True)
    rd = _make_run(batch, "run_mismatch", "HP_DEV_ONLY", 99)
    entry = _manifest_entry("run_mismatch", "HP_OFF", 99, rd)
    (batch / "manifest.json").write_text(json.dumps([entry]))

    with pytest.warns(UserWarning):
        runs = load_runs.discover_runs(tmp_path / "experiments")

    assert runs[0]["condition"] == "HP_DEV_ONLY"


# ─── 4. Fitness series shapes and generation order ────────────────────────────

def test_best_fitness_series_shape(four_run_experiments):
    run = load_runs.discover_runs(four_run_experiments)[0]
    series = load_runs.best_fitness_series(run)
    assert series.shape == (N_GENS,)
    assert np.issubdtype(series.dtype, np.floating)


def test_mean_fitness_series_shape(four_run_experiments):
    run = load_runs.discover_runs(four_run_experiments)[0]
    series = load_runs.mean_fitness_series(run)
    assert series.shape == (N_GENS,)


def test_population_fitnesses_shape(four_run_experiments):
    run = load_runs.discover_runs(four_run_experiments)[0]
    pop = load_runs.population_fitnesses(run)
    assert pop.shape == (N_GENS, POP_SIZE)


def test_fitness_series_matches_npz_files(four_run_experiments):
    run = load_runs.discover_runs(four_run_experiments)[0]
    history_dir = run["run_dir"] / "history"
    files = sorted(history_dir.glob("gen_*.npz"), key=lambda p: int(p.stem[4:]))

    expected_best = []
    expected_mean = []
    expected_pop = []
    for f in files:
        with np.load(f) as data:
            expected_best.append(float(data["best_fitness"]))
            expected_mean.append(float(data["mean_fitness"]))
            expected_pop.append(data["fitnesses"].copy())

    np.testing.assert_array_equal(load_runs.best_fitness_series(run), expected_best)
    np.testing.assert_array_equal(load_runs.mean_fitness_series(run), expected_mean)
    np.testing.assert_array_equal(load_runs.population_fitnesses(run), np.stack(expected_pop))


def test_best_fitness_ge_mean_fitness(four_run_experiments):
    run = load_runs.discover_runs(four_run_experiments)[0]
    best = load_runs.best_fitness_series(run)
    mean = load_runs.mean_fitness_series(run)
    assert np.all(best >= mean - 1e-12)


# ─── 5. Unequal run counts handled without crashing ──────────────────────────

def test_unequal_run_counts_no_crash(tmp_path):
    batch = tmp_path / "experiments" / "batch_c"
    batch.mkdir(parents=True)
    entries = []
    for i in range(2):
        rd = _make_run(batch, f"run_off_{i}", "HP_OFF", 50 + i)
        entries.append(_manifest_entry(f"run_off_{i}", "HP_OFF", 50 + i, rd))
    (batch / "manifest.json").write_text(json.dumps(entries))

    grouped = load_runs.runs_by_condition(tmp_path / "experiments")
    assert len(grouped["HP_OFF"]) == 2
    assert len(grouped["HP_DEV_ONLY"]) == 0
    assert len(grouped["HP_BEHAVIOUR_ONLY"]) == 0
    assert len(grouped["HP_BOTH"]) == 0


def test_single_run_condition_no_crash(tmp_path):
    batch = tmp_path / "experiments" / "batch_d"
    batch.mkdir(parents=True)
    rd = _make_run(batch, "run_both_1", "HP_BOTH", 77)
    (batch / "manifest.json").write_text(
        json.dumps([_manifest_entry("run_both_1", "HP_BOTH", 77, rd)])
    )
    grouped = load_runs.runs_by_condition(tmp_path / "experiments")
    assert len(grouped["HP_BOTH"]) == 1
    pop = load_runs.population_fitnesses(grouped["HP_BOTH"][0])
    assert pop.shape == (N_GENS, POP_SIZE)
