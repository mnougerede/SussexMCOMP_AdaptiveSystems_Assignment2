import numpy as np


class GA:
    def __init__(self, pop_size, genotype_length, p_m=0.03, sigma_m=0.1, elite_n=1, tournament_k=3, rng=None):
        self.pop_size = pop_size
        self.genotype_length = genotype_length
        self.p_m = p_m
        self.sigma_m = sigma_m
        self.elite_n = elite_n
        self.tournament_k = tournament_k
        self.rng = rng if rng is not None else np.random.default_rng()

    def initial_population(self):
        return self.rng.uniform(-1.0, 1.0, size=(self.pop_size, self.genotype_length))

    def _tournament_select(self, population, fitness_scores):
        indices = self.rng.choice(len(population), size=self.tournament_k, replace=False)
        winner = indices[np.argmax(fitness_scores[indices])]
        return population[winner]

    def _mutate(self, individual):
        mask = self.rng.random(self.genotype_length) < self.p_m
        noise = self.rng.normal(0.0, self.sigma_m, self.genotype_length)
        individual = individual + mask * noise
        # Iterative boundary reflection
        while True:
            above = individual > 1.0
            below = individual < -1.0
            if not (above.any() or below.any()):
                break
            individual[above] = 2.0 - individual[above]
            individual[below] = -2.0 - individual[below]
        return individual

    def step(self, population, fitness_scores):
        new_pop = np.empty_like(population)
        elite_indices = np.argsort(fitness_scores)[-self.elite_n:][::-1]
        new_pop[:self.elite_n] = population[elite_indices]

        for i in range(self.elite_n, self.pop_size):
            parent = self._tournament_select(population, fitness_scores)
            new_pop[i] = self._mutate(parent.copy())

        return new_pop
