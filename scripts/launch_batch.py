"""Batch launcher for GA experiments.

Example:
    python scripts/launch_batch.py \\
        --batch pilot_01 \\
        --conditions no_hp dev_only \\
        --n_runs 2 --base_seed 100 --n_workers 4
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from experiments.config import Condition, GAConfig, RunConfig
from experiments.evolve import run_experiment
from experiments.io import load_best_history, update_run_entry

REPO_ROOT = Path(__file__).parent.parent
TARGETS_FILE = REPO_ROOT / "plan" / "experiment_targets.json"

# Friendly CLI aliases for condition names
_ALIASES: dict[str, str] = {
    "no_hp": "HP_OFF",
    "dev_only": "HP_DEV_ONLY",
    "behaviour_only": "HP_BEHAVIOUR_ONLY",
    "both": "HP_BOTH",
}


def _resolve_condition(name: str) -> Condition:
    canonical = _ALIASES.get(name.lower(), name.upper())
    try:
        return Condition[canonical]
    except KeyError:
        valid = list(_ALIASES.keys()) + [c.name for c in Condition]
        raise SystemExit(f"Unknown condition {name!r}. Valid values: {valid}")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_targets() -> dict:
    if TARGETS_FILE.exists():
        with open(TARGETS_FILE) as f:
            return json.load(f)
    return {"n_gens": 200, "n_runs_per_condition": 5}


def main() -> None:
    targets = _load_targets()

    parser = argparse.ArgumentParser(description="Launch a labelled batch of GA experiments.")
    parser.add_argument("--batch", required=True, help="Batch label (used for output dir and batch file name)")
    parser.add_argument("--conditions", nargs="+", required=True, metavar="COND",
                        help="Condition names: no_hp, dev_only, behaviour_only, both, or HP_* enum names")
    parser.add_argument("--n_runs", type=int, required=True, help="Number of runs per condition")
    parser.add_argument("--base_seed", type=int, required=True,
                        help="Seed formula: base_seed + condition_index * 100 + run_index")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Worker processes for parallel fitness evaluation within each run")
    parser.add_argument("--n_gens", type=int, default=targets["n_gens"],
                        help=f"Generations per run (default: {targets['n_gens']} from experiment_targets.json)")
    parser.add_argument("--pop_size", type=int, default=None,
                        help="Population size (default: GAConfig default)")
    args = parser.parse_args()

    conditions = [_resolve_condition(c) for c in args.conditions]

    # Build the full list of (condition, seed) pairs
    run_specs = [
        {"condition": cond.name, "seed": args.base_seed + ci * 100 + ri}
        for ci, cond in enumerate(conditions)
        for ri in range(args.n_runs)
    ]

    # Write initial batch file
    batches_dir = REPO_ROOT / "batches"
    batches_dir.mkdir(exist_ok=True)
    batch_file = batches_dir / f"{args.batch}.json"

    batch_record: dict = {
        "label": args.batch,
        "created_at": _iso_now(),
        "conditions": [c.name for c in conditions],
        "n_runs": args.n_runs,
        "base_seed": args.base_seed,
        "n_workers": args.n_workers,
        "n_gens": args.n_gens,
        "runs": [
            {
                "condition": r["condition"],
                "seed": r["seed"],
                "status": "pending",
                "output_dir": None,
                "completed_at": None,
                "final_best_fitness": None,
            }
            for r in run_specs
        ],
        "completed_at": None,
    }
    with open(batch_file, "w") as f:
        json.dump(batch_record, f, indent=2)
    print(f"Batch '{args.batch}': {len(run_specs)} run(s) queued → {batch_file}")

    manifest_path = REPO_ROOT / "experiments" / args.batch / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Launch runs sequentially
    for idx, spec in enumerate(run_specs):
        condition = Condition[spec["condition"]]
        seed = spec["seed"]
        output_dir = REPO_ROOT / "experiments" / args.batch / f"{condition.name.lower()}_s{seed}"

        ga_kwargs: dict = {"n_gens": args.n_gens}
        if args.pop_size is not None:
            ga_kwargs["pop_size"] = args.pop_size

        config = RunConfig(
            ga=GAConfig(**ga_kwargs),
            condition=condition,
            seed=seed,
            output_dir=str(output_dir),
            n_workers=args.n_workers,
        )

        print(f"[{idx + 1}/{len(run_specs)}] {condition.name} seed={seed} ...", flush=True)
        run_experiment(config, manifest_path)

        # Tag the manifest entry with the batch label so the status script can find it
        update_run_entry(manifest_path, output_dir.name, batch=args.batch)

        # Get the final best fitness from the last best_per_gen record
        bh = load_best_history(str(output_dir))
        final_fitness: float | None = (
            float(bh["fitness"][-1]) if "fitness" in bh and len(bh["fitness"]) > 0 else None
        )

        # Update batch file for this run
        with open(batch_file) as f:
            record = json.load(f)
        record["runs"][idx].update({
            "status": "complete",
            "output_dir": str(output_dir),
            "completed_at": _iso_now(),
            "final_best_fitness": final_fitness,
        })
        with open(batch_file, "w") as f:
            json.dump(record, f, indent=2)

        fitness_str = f"{final_fitness:.4f}" if final_fitness is not None else "n/a"
        print(f"    done — best fitness: {fitness_str}")

    # Mark entire batch complete
    with open(batch_file) as f:
        record = json.load(f)
    record["completed_at"] = _iso_now()
    with open(batch_file, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\nBatch '{args.batch}' complete.")
    print(f"  Runs:      experiments/{args.batch}/")
    print(f"  Batch file: {batch_file}")


if __name__ == "__main__":
    main()
