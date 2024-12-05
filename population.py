from vector import Vector
import numpy as np
import heapq


class Population:
    __MUTATION_NORMAL_DEVIATION = 0.3
    __ELITE_PERCENTAGE = 0.1
    __LOWER_POTENTIAL_PERCENTAGE = 0.1
    __MATING_POOL_PERCENTAGE = 0.8
    __TOURNAMENT_PERCENTAGE = 0.06

    def __init__(self, num_individuals: int, num_iteration: int):
        self.__num_iteration = num_iteration
        self.__num_individuals = num_individuals
        self.__population: list[Vector] = [Population.__creating_individual() for _ in range(num_individuals)]

    @staticmethod
    def __creating_individual() -> Vector:
        random_vector = np.random.uniform(size=Vector.len())
        return Vector(random_vector)

    def __selection(self) -> list[Vector]:
        total_num = self.__num_individuals
        mating_pool: list[Vector] = []
        for i in range(int(total_num * Population.__MATING_POOL_PERCENTAGE)):
            sub_pool_indices = (np.random.uniform(size=int(total_num * Population.__TOURNAMENT_PERCENTAGE))
                                * np.random.randint(low=total_num, high=3 * total_num, size=1)) % total_num
            sub_pool = [self.__population[j] for j in sub_pool_indices]

            heapq.heapify(sub_pool)
            mating_pool.append(heapq.heappop(sub_pool))

        return mating_pool

    @staticmethod
    def __uniform_cross_over(vector1: Vector, vector2: Vector) -> list[Vector]:
        lookup_dict = {0: vector1.values, 1: vector2.values}
        children: list[list] = [[], []]

        for i in range(Vector.len()):
            random_num = np.random.randint(low=0, high=2, size=1)  # it is one or zero
            for j in range(2):
                children[j].append(lookup_dict.get((random_num + j) % 2)[i])

        return [Vector(np.array(child)) for child in children]

    @staticmethod
    def __mutate(vector: Vector) -> Vector:
        values = vector.values
        num_mutations = np.ceil(abs(np.random.normal(size=1)))
        mutation_value = np.random.normal(scale=Population.__MUTATION_NORMAL_DEVIATION, size=1)
        for i in range(num_mutations):
            rand_int = np.random.randint(low=0, high=Vector.len(), size=1)
            values[rand_int] += mutation_value

        return Vector(values)

    def genetic_algorithm(self) -> list[int]:
        for i in range(self.__num_iteration):
            current_population: list[Vector] = self.__population.copy()

            Vector.max_comparison(True)
            heapq.heapify(current_population)
            elite_individuals: list[Vector] = []
            for j in range(int(self.__num_individuals * Population.__ELITE_PERCENTAGE)):
                elite_individuals.append(heapq.heappop(current_population))

            Vector.max_comparison(False)
            heapq.heapify(current_population)
            low_potential_individuals: list[Vector] = []
            for j in range(int(self.__num_individuals * Population.__LOWER_POTENTIAL_PERCENTAGE)):
                low_potential_individuals.append(heapq.heappop(current_population))

            mating_pool = self.__selection()
