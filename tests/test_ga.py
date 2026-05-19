import numpy as np
import pytest
from src.ga.ga import GA


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_initial_population_shape(rng):
    ga = GA(pop_size=20, genotype_length=10, rng=rng)
    pop = ga.initial_population()
    assert pop.shape == (20, 10)


def test_initial_population_range(rng):
    ga = GA(pop_size=100, genotype_length=50, rng=rng)
    pop = ga.initial_population()
    assert np.all(pop >= -1.0) and np.all(pop <= 1.0)


def test_elite_preserved(rng):
    ga = GA(pop_size=10, genotype_length=5, elite_n=2, rng=rng)
    pop = ga.initial_population()
    fitness = rng.random(10)
    elite_indices = np.argsort(fitness)[-2:]
    elite_before = pop[elite_indices]

    new_pop = ga.step(pop, fitness)

    # Both elite individuals must appear unchanged somewhere in new_pop
    for elite in elite_before:
        assert any(np.array_equal(elite, row) for row in new_pop)


def test_tournament_biased_toward_best(rng):
    ga = GA(pop_size=10, genotype_length=3, tournament_k=3, rng=rng)
    pop = ga.initial_population()
    fitness = np.arange(10, dtype=float)  # individual 9 is best

    wins = 0
    trials = 1000
    for _ in range(trials):
        selected = ga._tournament_select(pop, fitness)
        if np.array_equal(selected, pop[9]):
            wins += 1

    # Expected win rate for best in tournament of k=3 from n=10 is ~P(9 in sample)=3/10=0.3
    assert wins > trials * 0.15, f"Best individual won only {wins}/{trials} times"


def test_mutation_p1_changes_all(rng):
    ga = GA(pop_size=5, genotype_length=20, p_m=1.0, sigma_m=0.1, rng=rng)
    original = np.zeros(20)
    mutated = ga._mutate(original.copy())
    assert not np.array_equal(original, mutated)
    assert np.any(mutated != 0.0)


def test_mutation_p0_changes_nothing(rng):
    ga = GA(pop_size=5, genotype_length=20, p_m=0.0, sigma_m=0.1, rng=rng)
    original = np.zeros(20)
    mutated = ga._mutate(original.copy())
    assert np.array_equal(original, mutated)


def test_mutation_stays_in_bounds_stress(rng):
    ga = GA(pop_size=50, genotype_length=100, p_m=1.0, sigma_m=2.0, rng=rng)
    pop = ga.initial_population()
    fitness = rng.random(50)
    new_pop = ga.step(pop, fitness)
    assert np.all(new_pop >= -1.0) and np.all(new_pop <= 1.0)


def test_step_preserves_shape(rng):
    ga = GA(pop_size=15, genotype_length=8, rng=rng)
    pop = ga.initial_population()
    fitness = rng.random(15)
    new_pop = ga.step(pop, fitness)
    assert new_pop.shape == (15, 8)
