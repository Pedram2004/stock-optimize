from vector import Vector
from base_optimizer import Optimizer
import numpy as np
import matplotlib.pyplot as plt
import heapq


class GeneticAlgorithmOptimizer(Optimizer):
    __MUTATION_NORMAL_DEVIATION = 0.3
    __MUTATION_PERCENTAGE = 0.03
    __ELITE_PERCENTAGE = 0.1
    __LOWER_POTENTIAL_PERCENTAGE = 0.1
    __MATING_POOL_PERCENTAGE = 0.8
    __TOURNAMENT_PERCENTAGE = 0.06

    def __init__(self, num_individuals: int, num_iterations: int, random_state: int = 42):
        super().__init__(num_iterations, random_state)
        self.__num_individuals = num_individuals
        print(self._num_iterations)
        self.__population: list[Vector] = [GeneticAlgorithmOptimizer.__create_individual() for _ in range(num_individuals)]

    @staticmethod
    def __create_individual() -> Vector:
        random_vector = np.random.uniform(size=Vector.len())
        return Vector(random_vector)

    def __selection(self) -> list[Vector]:
        total_num = self.__num_individuals
        mating_pool: list[Vector] = []
        for i in range(int(total_num * GeneticAlgorithmOptimizer.__MATING_POOL_PERCENTAGE)):
            sub_pool_indices = (np.random.uniform(size=int(total_num * GeneticAlgorithmOptimizer.__TOURNAMENT_PERCENTAGE))
                                * np.random.randint(low=total_num, high=3 * total_num, size=1)) % total_num
            sub_pool = [self.__population[j] for j in sub_pool_indices]

            heapq.heapify(sub_pool)
            mating_pool.append(heapq.heappop(sub_pool))

        return mating_pool

    @staticmethod
    def __uniform_cross_over(vector_pair: list[Vector]) -> list[Vector]:
        lookup_dict = {i: vector for i, vector in enumerate(vector_pair)}
        children: list[list] = [[], []]

        for i in range(Vector.len()):
            random_num = np.random.randint(low=0, high=2, size=1)  # it is one or zero
            for j in range(2):
                children[j].append(lookup_dict.get((random_num + j) % 2).values[i])

        return [Vector(np.array(child)) for child in children]

    @staticmethod
    def __mutate(vector: Vector) -> Vector:
        values = vector.values
        num_mutations = np.ceil(abs(np.random.normal(size=1)))
        mutation_value = np.random.normal(scale=GeneticAlgorithmOptimizer.__MUTATION_NORMAL_DEVIATION, size=1)
        for i in range(num_mutations):
            rand_int = np.random.randint(low=0, high=Vector.len(), size=1)
            values[rand_int] += mutation_value

        return Vector(values)

    def optimize(self) -> list[float]:
        highest_fitness_chromosomes: list[float] = []
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
                    (1 - (GeneticAlgorithmOptimizer.__LOWER_POTENTIAL_PERCENTAGE + GeneticAlgorithmOptimizer.__ELITE_PERCENTAGE)) / 2)
            for _ in range(int(self.__num_individuals * new_generation_percentage)):
                parents_indices = [k for k in np.random.randint(low=0, high=int(
                    self.__num_individuals * GeneticAlgorithmOptimizer.__MATING_POOL_PERCENTAGE), size=2)]
                new_generation.extend(GeneticAlgorithmOptimizer.__uniform_cross_over([mating_pool[k] for k in parents_indices]))

            for _ in range(int(self.__num_individuals * GeneticAlgorithmOptimizer.__MUTATION_PERCENTAGE)):
                random_index = np.random.randint(low=0, high=self.__num_individuals + 1, size=1)
                new_generation[random_index] = GeneticAlgorithmOptimizer.__mutate(new_generation[random_index])

            self.__population = new_generation
            highest_fitness_chromosomes.append((max(self.__population)).fitness)

        return highest_fitness_chromosomes
    
    def draw_chart(self) -> None:
        pass


class BeamSearchOptimizer(Optimizer):
    def __init__(self, beam_length: int, num_iterations: int, learning_rate: float = 0.1, random_state: int = 42):
        super().__init__(num_iterations, random_state)
        self.__beam_length = beam_length
        self.__beam = [Vector(np.random.uniform(size=Vector.len())) for _ in range(self.__beam_length)]
        self.__LEARNING_RATE = learning_rate

    def optimize(self) -> Vector:
        for _ in range(self._num_iterations):
            neighbors = []
            for vector in self.__beam:
                neighbors.extend(vector.get_neighbors(self.__LEARNING_RATE))
            best_neighbors = heapq.nlargest(self.__beam_length, neighbors, key=lambda x: x.fitness) # TODO: check with pedram if we should account for the case where the beam length is less than log of the number of neighbors
            self.__beam = best_neighbors
        return max(self.__beam, key=lambda x: x.fitness)

    def draw_chart(self) -> None:
        pass


class RandomBeamSearchOptimizer(Optimizer):
    def __init__(self, beam_length: int, num_iterations: int, learning_rate: float = 0.1, random_state: int = 42):
        super().__init__(num_iterations, random_state)
        self.__beam_length = beam_length
        self.__beam = [Vector(np.random.uniform(size=Vector.len())) for _ in range(self.__beam_length)]
        self.__LEARNING_RATE = learning_rate

    def optimize(self) -> Vector:
        for _ in range(self._num_iterations):
            neighbors = []
            for vector in self.__beam:
                neighbors.extend(vector.get_neighbors(self.__LEARNING_RATE))
            prob = np.array([vector.fitness for vector in neighbors])
            prob /= prob.sum()
            best_neighbors = np.random.choice(neighbors, size=self.__beam_length, replace=False, p=prob)
            self.__beam = best_neighbors
        return max(self.__beam, key=lambda x: x.fitness)

    def draw_chart(self) -> None:
        pass


class SimulatedAnnealingOptimizer(Optimizer):
    def __init__(self, num_iterations: int, learning_rate: float = 0.1, random_state: int = 42):
        super().__init__(num_iterations, random_state)
        self.__current_state = Vector(np.random.uniform(size=Vector.len()))
        self.__neighbours = (self.__current_state, self.__current_state.get_neighbors()) #TODO radius unfilled
        self.__LEARNING_RATE = learning_rate

    def __perturbate(self) -> Vector:
        if self.__current_state != self.__neighbours[0]:
            self.__neighbours = (self.__current_state, self.__current_state.get_neighbors(radius=self.__LEARNING_RATE))
        new_state = np.random.choice(self.__neighbours[1])
        self.__neighbours[1].remove(new_state)
        return new_state

    def __accept_change(self, new_state: Vector, temperature: int) -> bool:
        delta_e = new_state.fitness - self.__current_state.fitness
        if delta_e < 0:
            return True
        else:
            acceptance_probability = np.exp(-delta_e / temperature)
            if np.random.uniform() <= acceptance_probability:
                return True
            
        return False

    def optimize(self) -> Vector:
        for temperature in range(self._num_iterations, 0, -1):
            new_state = self.__perturbate()
            if self.__accept_change(new_state, temperature):
                self.__current_state = new_state
        return self.__current_state

    def draw_chart(self) -> None:
        pass