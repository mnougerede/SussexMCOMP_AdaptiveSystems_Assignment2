import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


# ─── Atomic writes ────────────────────────────────────────────────────────────

def atomic_write_json(path: str | Path, data) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def atomic_write_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    # Keep .npz suffix so numpy doesn't append a second one
    tmp = path.with_stem(path.stem + ".tmp")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ─── Manifest ─────────────────────────────────────────────────────────────────

def load_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def save_manifest(path: str | Path, entries: list[dict]) -> None:
    atomic_write_json(path, entries)


def register_run(
    manifest_path: str | Path,
    run_id: str,
    condition: str,
    seed: int,
    output_dir: str,
) -> None:
    entries = load_manifest(manifest_path)
    for entry in entries:
        if entry["run_id"] == run_id:
            return
    entries.append({
        "run_id": run_id,
        "condition": condition,
        "seed": seed,
        "status": "pending",
        "started_at": None,
        "completed_at": None,
        "current_gen": 0,
        "output_dir": output_dir,
    })
    save_manifest(manifest_path, entries)


def update_run_entry(manifest_path: str | Path, run_id: str, **kwargs) -> None:
    entries = load_manifest(manifest_path)
    for entry in entries:
        if entry["run_id"] == run_id:
            entry.update(kwargs)
            break
    save_manifest(manifest_path, entries)


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(
    output_dir: str | Path,
    generation: int,
    population: np.ndarray,
    fitnesses: np.ndarray,
    rng_state: dict,
    elapsed_seconds: float,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_npz(
        output_dir / "checkpoint.npz",
        population=population,
        fitnesses=fitnesses,
        generation=np.array(generation),
        elapsed_seconds=np.array(elapsed_seconds),
    )
    atomic_write_json(output_dir / "checkpoint.rng.json", rng_state)


def load_checkpoint(output_dir: str | Path) -> Optional[dict]:
    output_dir = Path(output_dir)
    npz_path = output_dir / "checkpoint.npz"
    rng_path = output_dir / "checkpoint.rng.json"
    if not npz_path.exists() or not rng_path.exists():
        return None
    with np.load(npz_path) as data:
        checkpoint = {
            "generation": int(data["generation"]),
            "population": data["population"].copy(),
            "fitnesses": data["fitnesses"].copy(),
            "elapsed_seconds": float(data["elapsed_seconds"]),
        }
    with open(rng_path) as f:
        checkpoint["rng_state"] = json.load(f)
    return checkpoint


# ─── Per-generation history ───────────────────────────────────────────────────

def append_generation_record(
    output_dir: str | Path,
    generation: int,
    **arrays: np.ndarray,
) -> None:
    history_dir = Path(output_dir) / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_npz(history_dir / f"gen_{generation:04d}.npz", **arrays)


def load_history(output_dir: str | Path) -> dict[str, np.ndarray]:
    history_dir = Path(output_dir) / "history"
    if not history_dir.exists():
        return {}
    files = sorted(history_dir.glob("gen_*.npz"))
    if not files:
        return {}
    records = []
    for f in files:
        with np.load(f) as data:
            records.append({k: data[k].copy() for k in data})
    keys = records[0].keys()
    return {k: np.stack([r[k] for r in records]) for k in keys}


# ─── Best per generation ──────────────────────────────────────────────────────

def save_best_for_generation(
    output_dir: str | Path,
    generation: int,
    **arrays: np.ndarray,
) -> None:
    best_dir = Path(output_dir) / "best_per_gen"
    best_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_npz(best_dir / f"gen_{generation:04d}.npz", **arrays)


def load_best_history(output_dir: str | Path) -> dict[str, np.ndarray]:
    best_dir = Path(output_dir) / "best_per_gen"
    if not best_dir.exists():
        return {}
    files = sorted(best_dir.glob("gen_*.npz"))
    if not files:
        return {}
    records = []
    for f in files:
        with np.load(f) as data:
            records.append({k: data[k].copy() for k in data})
    keys = records[0].keys()
    return {k: np.stack([r[k] for r in records]) for k in keys}


# ─── Git commit ───────────────────────────────────────────────────────────────

def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return ""
