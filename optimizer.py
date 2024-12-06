from vector import Vector
from base_optimizer import Optimizer
import numpy as np
import matplotlib.pyplot as plt
import heapq


class GeneticAlgorithmOptimizer(Optimizer):
    __MUTATION_NORMAL_DEVIATION = 0.1
    __MUTATION_PERCENTAGE = 0.03
    __ELITE_PERCENTAGE = 0.1
    __LOWER_POTENTIAL_PERCENTAGE = 0.05
    __MATING_POOL_PERCENTAGE = 0.85
    __TOURNAMENT_PERCENTAGE = 0.05

    def __init__(self, num_individuals: int, num_iterations: int, random_state: int = 1002):
        super().__init__(num_iterations, random_state)
        self.__num_individuals = num_individuals
        self.__population: list[Vector] = [GeneticAlgorithmOptimizer.__create_individual() for _ in
                                           range(num_individuals)]
        self.__iterations = []

    @staticmethod
    def __create_individual() -> Vector:
        random_vector = np.random.uniform(size=Vector.len())
        return Vector(random_vector)

    def __selection(self) -> list[Vector]:
        total_num = self.__num_individuals
        mating_pool: list[Vector] = []
        for _ in range(int(total_num * GeneticAlgorithmOptimizer.__MATING_POOL_PERCENTAGE)):
            sub_pool_indices = np.ceil(
                np.random.uniform(size=int(total_num * GeneticAlgorithmOptimizer.__TOURNAMENT_PERCENTAGE))
                * np.random.randint(low=total_num, high=3 * total_num, size=1)) % (total_num - 1)
            sub_pool_indices = np.int64(sub_pool_indices)
            sub_pool = [self.__population[j] for j in list(sub_pool_indices)]
            heapq.heapify(sub_pool)
            mating_pool.append(heapq.heappop(sub_pool))

        return mating_pool

    @staticmethod
    def __uniform_cross_over(vector_pair: list[Vector]) -> list[Vector]:
        lookup_dict = {i: vector for i, vector in enumerate(vector_pair)}
        children: list[list] = [[], []]

        for i in range(Vector.len()):
            random_num = np.random.randint(low=0, high=2)  # it is one or zero
            for j in range(2):
                children[j].append(lookup_dict.get((random_num + j) % 2).values[i])

        return [Vector(np.array(child)) for child in children]

    @staticmethod
    def __mutate(vector: Vector) -> Vector:
        values = vector.values
        num_mutations = np.ceil(abs(np.random.normal()))
        mutation_value = np.random.normal(scale=GeneticAlgorithmOptimizer.__MUTATION_NORMAL_DEVIATION)
        for i in range(int(num_mutations)):
            rand_int = np.random.randint(low=0, high=Vector.len(), size=1)
            values[rand_int] += abs(mutation_value)

        return Vector(values)

    def optimize(self) -> Vector:
        for _ in range(self._num_iterations):
            new_generation: list[Vector] = []
            current_population: list[Vector] = self.__population.copy()

            Vector.max_comparison(True)
            heapq.heapify(current_population)
            for _ in range(int(self.__num_individuals * GeneticAlgorithmOptimizer.__ELITE_PERCENTAGE)):
                new_generation.append(heapq.heappop(current_population))

            Vector.max_comparison(False)
            heapq.heapify(current_population)
            for _ in range(int(self.__num_individuals * GeneticAlgorithmOptimizer.__LOWER_POTENTIAL_PERCENTAGE)):
                new_generation.append(heapq.heappop(current_population))

            mating_pool = self.__selection()
            new_generation_percentage = (
                    (1 - (GeneticAlgorithmOptimizer.__LOWER_POTENTIAL_PERCENTAGE +
                          GeneticAlgorithmOptimizer.__ELITE_PERCENTAGE)) / 2)
            for _ in range(int(self.__num_individuals * new_generation_percentage)):
                parents_indices = np.random.randint(low=0, high=int(
                    self.__num_individuals * GeneticAlgorithmOptimizer.__MATING_POOL_PERCENTAGE), size=2)
                new_generation.extend(
                    GeneticAlgorithmOptimizer.__uniform_cross_over([mating_pool[k] for k in parents_indices]))

            for _ in range(int(self.__num_individuals * GeneticAlgorithmOptimizer.__MUTATION_PERCENTAGE)):
                random_index = np.random.randint(low=0, high=(self.__num_individuals - 1))
                new_generation[random_index] = GeneticAlgorithmOptimizer.__mutate(new_generation[random_index])

            self.__population = new_generation

            fittest = max(self.__population)
            self.__iterations.append(fittest.fitness)

        return fittest
    
    def get_plot_info(self) -> tuple[range, list[float]]:
        return range(self._num_iterations), self.__iterations


class BeamSearchOptimizer(Optimizer):
    def __init__(self, beam_length: int, num_iterations: int, learning_rate: float = 0.1, random_state: int = 42):
        super().__init__(num_iterations, random_state)
        self.__beam_length = beam_length
        self.__beam = [Vector(np.random.uniform(size=Vector.len())) for _ in range(self.__beam_length)]
        self.__LEARNING_RATE = learning_rate
        self.__iterations = []

    def optimize(self) -> Vector:
        for _ in range(self._num_iterations):
            neighbors = []
            for vector in self.__beam:
                neighbors.extend(vector.get_neighbours(self.__LEARNING_RATE))
            best_neighbors = heapq.nlargest(self.__beam_length, neighbors, key=lambda
                x: x.fitness)
            self.__beam = best_neighbors

            self.__iterations.append(max(self.__beam).fitness)
        return max(self.__beam, key=lambda x: x.fitness)

    def get_plot_info(self) -> tuple[range, list[float]]:
        return range(self._num_iterations), self.__iterations


class RandomBeamSearchOptimizer(Optimizer):
    def __init__(self, beam_length: int, num_iterations: int, learning_rate: float = 0.1, random_state: int = 42):
        super().__init__(num_iterations, random_state)
        self.__beam_length = beam_length
        self.__beam = [Vector(np.random.uniform(size=Vector.len())) for _ in range(self.__beam_length)]
        self.__LEARNING_RATE = learning_rate
        self.__iterations = []

    def optimize(self) -> Vector:
        for _ in range(self._num_iterations):
            neighbors = []
            for vector in self.__beam:
                neighbors.extend(vector.get_neighbours(self.__LEARNING_RATE))
            prob = np.array([vector.fitness for vector in neighbors])
            prob /= prob.sum()
            best_neighbors = np.random.choice(neighbors, size=self.__beam_length, replace=False, p=prob)
            self.__beam = best_neighbors

            self.__iterations.append(max(self.__beam).fitness)
        return max(self.__beam, key=lambda x: x.fitness)

    def get_plot_info(self) -> tuple[range, list[float]]:
        return range(self._num_iterations), self.__iterations


class SimulatedAnnealingOptimizer(Optimizer):
    def __init__(self, num_iterations: int, learning_rate: float = 0.1, random_state: int = 42):
        super().__init__(num_iterations, random_state)
        self.__current_state = Vector(np.random.uniform(size=Vector.len()))
        self.__LEARNING_RATE = learning_rate
        self.__neighbours = (self.__current_state, self.__current_state.get_neighbours(self.__LEARNING_RATE))
        self.__iterations = []

    def __perturbate(self) -> Vector:
        if self.__current_state != self.__neighbours[0] or not self.__neighbours[1]:
            self.__neighbours = (self.__current_state, self.__current_state.get_neighbours(self.__LEARNING_RATE))
        new_state = np.random.choice(self.__neighbours[1])
        self.__neighbours[1].remove(new_state)
        return new_state

    def __accept_change(self, new_state: Vector, temperature: float) -> bool:
        delta_e = new_state.fitness - self.__current_state.fitness
        if delta_e > 0:
            return True
        else:
            acceptance_probability = np.exp(delta_e / temperature)
            if np.random.uniform() <= acceptance_probability:
                return True

        return False

    def optimize(self) -> Vector:
        for temperature in range(self._num_iterations, 0, -1):
            new_state = self.__perturbate()
            if self.__accept_change(new_state, (temperature/self._num_iterations)**1.5):
                self.__current_state = new_state
            self.__iterations.append(self.__current_state.fitness)
        return self.__current_state

    
    def get_plot_info(self) -> tuple[range, list[float]]:
        return range(self._num_iterations), self.__iterations
    