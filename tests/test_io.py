import json
import os

import numpy as np
import pytest

from experiments.io import (
    append_generation_record,
    atomic_write_json,
    atomic_write_npz,
    load_best_history,
    load_checkpoint,
    load_history,
    load_manifest,
    register_run,
    save_best_for_generation,
    save_checkpoint,
    update_run_entry,
)


# ─── 1. Atomic write ──────────────────────────────────────────────────────────

def test_atomic_write_json_failure_leaves_original_untouched(tmp_path, monkeypatch):
    path = tmp_path / "data.json"
    path.write_text('{"original": true}')

    monkeypatch.setattr(os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError):
        atomic_write_json(path, {"new": "data"})

    assert json.loads(path.read_text()) == {"original": True}


def test_atomic_write_npz_failure_leaves_original_untouched(tmp_path, monkeypatch):
    path = tmp_path / "arrays.npz"
    np.savez(path, x=np.array([1, 2, 3]))

    monkeypatch.setattr(os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError):
        atomic_write_npz(path, x=np.array([9, 9, 9]))

    with np.load(path) as data:
        np.testing.assert_array_equal(data["x"], [1, 2, 3])


# ─── 2. Manifest ──────────────────────────────────────────────────────────────

def test_manifest_register_update_no_duplication(tmp_path):
    manifest_path = tmp_path / "manifest.json"

    register_run(manifest_path, "run_001", "HP_BOTH", 42, str(tmp_path / "run_001"))
    register_run(manifest_path, "run_002", "HP_OFF", 1, str(tmp_path / "run_002"))
    register_run(manifest_path, "run_001", "HP_BOTH", 42, str(tmp_path / "run_001"))  # duplicate

    entries = load_manifest(manifest_path)
    assert len(entries) == 2

    ids = [e["run_id"] for e in entries]
    assert ids.count("run_001") == 1

    update_run_entry(manifest_path, "run_001", status="running", current_gen=7)

    entries = load_manifest(manifest_path)
    run_001 = next(e for e in entries if e["run_id"] == "run_001")
    assert run_001["status"] == "running"
    assert run_001["current_gen"] == 7
    assert run_001["condition"] == "HP_BOTH"
    assert run_001["seed"] == 42

    run_002 = next(e for e in entries if e["run_id"] == "run_002")
    assert run_002["status"] == "pending"


# ─── 3. Checkpoint round-trip with RNG ────────────────────────────────────────

def test_checkpoint_rng_round_trip(tmp_path):
    rng = np.random.default_rng(99)
    population = rng.random((10, 35))
    fitnesses = rng.random(10)

    saved_state = rng.bit_generator.state
    expected_next = rng.random()  # draw to compare after restoration

    save_checkpoint(
        tmp_path,
        generation=5,
        population=population,
        fitnesses=fitnesses,
        rng_state=saved_state,
        elapsed_seconds=12.34,
    )

    checkpoint = load_checkpoint(tmp_path)

    assert checkpoint is not None
    assert checkpoint["generation"] == 5
    assert checkpoint["elapsed_seconds"] == pytest.approx(12.34)
    np.testing.assert_array_equal(checkpoint["population"], population)
    np.testing.assert_array_equal(checkpoint["fitnesses"], fitnesses)

    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = checkpoint["rng_state"]
    assert restored_rng.random() == expected_next


# ─── 4. History ───────────────────────────────────────────────────────────────

def test_history_append_and_load(tmp_path):
    for gen in range(4):
        fitnesses = np.arange(5, dtype=float) + gen * 10
        mean_fit = np.array(fitnesses.mean())
        append_generation_record(tmp_path, gen, fitnesses=fitnesses, mean_fitness=mean_fit)

    history = load_history(tmp_path)

    assert history["fitnesses"].shape == (4, 5)
    assert history["mean_fitness"].shape == (4,)

    np.testing.assert_array_equal(history["fitnesses"][0], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(history["fitnesses"][3], [30, 31, 32, 33, 34])
    assert history["mean_fitness"][0] == pytest.approx(2.0)
    assert history["mean_fitness"][3] == pytest.approx(32.0)


# ─── 5. Best per generation ───────────────────────────────────────────────────

def test_best_per_gen_save_and_load(tmp_path):
    for gen in range(3):
        genotype = np.ones(35) * gen
        fitness = np.array(float(gen * 5))
        save_best_for_generation(tmp_path, gen, genotype=genotype, fitness=fitness)

    best = load_best_history(tmp_path)

    assert best["genotype"].shape == (3, 35)
    assert best["fitness"].shape == (3,)

    np.testing.assert_array_equal(best["genotype"][0], np.zeros(35))
    np.testing.assert_array_equal(best["genotype"][2], np.ones(35) * 2)
    assert best["fitness"][1] == pytest.approx(5.0)


# ─── 6. Missing checkpoint returns None ───────────────────────────────────────

def test_missing_checkpoint_returns_none(tmp_path):
    assert load_checkpoint(tmp_path) is None


def test_partial_checkpoint_returns_none(tmp_path):
    # Only npz present, no rng sidecar
    np.savez(tmp_path / "checkpoint.npz", population=np.ones((5, 35)))
    assert load_checkpoint(tmp_path) is None
