import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from experiments.config import Condition, GAConfig, RunConfig, run_config_to_json
from experiments.io import (
    append_generation_record,
    get_git_commit,
    load_checkpoint,
    register_run,
    save_best_for_generation,
    save_checkpoint,
    update_run_entry,
)


def run_experiment(
    run_config: RunConfig,
    manifest_path: str | Path = "results/data/manifest.json",
) -> None:
    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = output_dir.name
    config_path = output_dir / "config.json"
    current_commit = get_git_commit()

    if config_path.exists():
        with open(config_path) as f:
            saved = json.load(f)
        saved_commit = saved.get("git_commit", "")
        if saved_commit and current_commit and saved_commit != current_commit:
            raise RuntimeError(
                f"Git commit mismatch: config has {saved_commit!r}, "
                f"current HEAD is {current_commit!r}"
            )
    else:
        run_config = dataclasses.replace(run_config, git_commit=current_commit)
        run_config_to_json(run_config, config_path)

    register_run(
        manifest_path,
        run_id=run_id,
        condition=run_config.condition.name,
        seed=run_config.seed,
        output_dir=str(output_dir),
    )
    update_run_entry(manifest_path, run_id, status="in_progress")

    checkpoint = load_checkpoint(output_dir)
    if checkpoint:
        start_gen = checkpoint["generation"] + 1
        rng = np.random.default_rng()
        rng.bit_generator.state = checkpoint["rng_state"]
        elapsed_seconds = checkpoint["elapsed_seconds"]
    else:
        start_gen = 0
        rng = np.random.default_rng(run_config.seed)
        elapsed_seconds = 0.0

    pop_size = run_config.ga.pop_size
    genotype_length = run_config.ctrnn.genotype_length
    t0 = time.monotonic()

    for gen in range(start_gen, run_config.ga.n_gens):
        # Stub: random population and fitnesses (real GA not yet wired)
        population = rng.random((pop_size, genotype_length))
        fitnesses = rng.random(pop_size)

        rng_state = rng.bit_generator.state
        current_elapsed = elapsed_seconds + (time.monotonic() - t0)

        save_checkpoint(
            output_dir,
            generation=gen,
            population=population,
            fitnesses=fitnesses,
            rng_state=rng_state,
            elapsed_seconds=current_elapsed,
        )

        best_idx = int(np.argmax(fitnesses))
        append_generation_record(
            output_dir, gen,
            fitnesses=fitnesses,
            best_fitness=np.array(fitnesses[best_idx]),
            mean_fitness=np.array(fitnesses.mean()),
        )
        save_best_for_generation(
            output_dir, gen,
            genotype=population[best_idx],
            fitness=np.array(fitnesses[best_idx]),
        )

        update_run_entry(manifest_path, run_id, current_gen=gen)

    update_run_entry(manifest_path, run_id, status="complete")


if __name__ == "__main__":
    from experiments.config import GAConfig

    config = RunConfig(
        ga=GAConfig(pop_size=10, n_gens=20, n_runs=1),
        condition=Condition.HP_OFF,
        seed=0,
        output_dir="results/data/test_run",
    )
    manifest = "results/data/manifest.json"
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    run_experiment(config, manifest)
    print("Done. Output in results/data/test_run/, manifest at", manifest)
