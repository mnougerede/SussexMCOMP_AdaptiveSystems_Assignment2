"""Read-only experiment status viewer.

Scans all manifest.json files under experiments/, aggregates run status against
targets in plan/experiment_targets.json, and prints a summary table.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
TARGETS_FILE = REPO_ROOT / "plan" / "experiment_targets.json"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _load_targets() -> dict:
    with open(TARGETS_FILE) as f:
        return json.load(f)


def _n_gens_for_run(output_dir: Path, fallback: int) -> int:
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("ga", {}).get("n_gens", fallback)
    return fallback


def _run_status(entry: dict, target_n_gens: int) -> str:
    """Returns 'done', 'running', or 'pending'."""
    if entry.get("status") == "complete":
        return "done"

    output_dir = Path(entry.get("output_dir", ""))
    if not output_dir.exists():
        return "pending"

    n_gens = _n_gens_for_run(output_dir, target_n_gens)

    history_dir = output_dir / "history"
    n_history = len(list(history_dir.glob("gen_*.npz"))) if history_dir.exists() else 0

    if n_history >= n_gens:
        return "done"
    if n_history > 0 or (output_dir / "checkpoint.npz").exists():
        return "running"
    return "pending"


def _final_best_fitness(output_dir: Path) -> float:
    best_dir = output_dir / "best_per_gen"
    if best_dir.exists():
        files = sorted(best_dir.glob("gen_*.npz"))
        if files:
            with np.load(files[-1]) as data:
                return float(data["fitness"])

    # Fallback: last history file
    history_dir = output_dir / "history"
    if history_dir.exists():
        files = sorted(history_dir.glob("gen_*.npz"))
        if files:
            with np.load(files[-1]) as data:
                if "best_fitness" in data:
                    return float(data["best_fitness"])

    # Fallback: checkpoint
    cp_path = output_dir / "checkpoint.npz"
    if cp_path.exists():
        with np.load(cp_path) as data:
            if "fitnesses" in data:
                return float(np.max(data["fitnesses"]))

    return float("nan")


def main() -> None:
    targets = _load_targets()
    target_runs = targets["n_runs_per_condition"]
    target_n_gens = targets["n_gens"]
    target_conditions = targets["conditions"]

    if not EXPERIMENTS_DIR.exists():
        print("No experiments/ directory found — nothing to report.")
        return

    all_entries: list[dict] = []
    batch_labels: set[str] = set()

    for manifest_path in sorted(EXPERIMENTS_DIR.rglob("manifest.json")):
        with open(manifest_path) as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError:
                continue
        for e in entries:
            all_entries.append(e)
            if "batch" in e and e["batch"]:
                batch_labels.add(e["batch"])

    # ── Status table ─────────────────────────────────────────────────────────
    # Collect all conditions seen, with target conditions first
    seen_conditions = {e.get("condition", "") for e in all_entries}
    all_conditions = target_conditions + sorted(
        c for c in seen_conditions if c and c not in target_conditions
    )

    counts: dict[str, dict] = {c: {"done": 0, "running": 0} for c in all_conditions}
    for entry in all_entries:
        cond = entry.get("condition", "")
        if not cond:
            continue
        if cond not in counts:
            counts[cond] = {"done": 0, "running": 0}
        s = _run_status(entry, target_n_gens)
        if s == "done":
            counts[cond]["done"] += 1
        elif s == "running":
            counts[cond]["running"] += 1

    col = (28, 8, 8, 10, 12)
    header = (
        f"{'Condition':<{col[0]}} {'Target':>{col[1]}} {'Done':>{col[2]}}"
        f" {'Running':>{col[3]}} {'Remaining':>{col[4]}}"
    )
    sep = "-" * sum(col) + "-" * (len(col) - 1)

    print()
    print(header)
    print(sep)

    total_target = 0
    total_done = 0
    for cond in all_conditions:
        c = counts.get(cond, {"done": 0, "running": 0})
        done = c["done"]
        running = c["running"]
        remaining = max(0, target_runs - done)
        total_target += target_runs
        total_done += done
        print(
            f"{cond:<{col[0]}} {target_runs:>{col[1]}} {done:>{col[2]}}"
            f" {running:>{col[3]}} {remaining:>{col[4]}}"
        )

    print(sep)
    print(f"\nTotal: {total_done} / {total_target} runs complete\n")

    # ── Completed runs list ───────────────────────────────────────────────────
    completed = []
    for entry in all_entries:
        if _run_status(entry, target_n_gens) != "done":
            continue
        output_dir = Path(entry.get("output_dir", ""))
        fitness = _final_best_fitness(output_dir) if output_dir.exists() else float("nan")
        completed.append({
            "condition": entry.get("condition", ""),
            "seed": entry.get("seed", ""),
            "fitness": fitness,
            "completed_at": entry.get("completed_at") or "",
        })

    completed.sort(key=lambda x: (x["condition"], x["completed_at"]))

    if completed:
        hdr2 = f"{'Condition':<28} {'Seed':>8} {'Best Fitness':>14}  Completed At"
        print(hdr2)
        print("-" * len(hdr2))
        for r in completed:
            print(
                f"{r['condition']:<28} {str(r['seed']):>8}"
                f" {r['fitness']:>14.6f}  {r['completed_at']}"
            )
        print()

    # ── Batch labels ─────────────────────────────────────────────────────────
    if batch_labels:
        print("Batch labels:", ", ".join(sorted(batch_labels)))
    else:
        print("No batch labels found.")


if __name__ == "__main__":
    main()
