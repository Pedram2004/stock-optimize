from vector import Vector
import numpy as np


class Population:
    def __init__(self, num_individuals: int, num_iteration: int):
        self.__num_iteration = num_iteration
        self.__population: list[Vector] = [Population.__creating_individual() for _ in range(num_individuals)]

    @staticmethod
    def __creating_individual() -> Vector:
        random_vector = np.random.uniform(size=Vector.len())
        return Vector(random_vector)

    @staticmethod
    def __cross_over(vector1: Vector, vector2: Vector) -> list[Vector]:
        lookup_dict = {0: vector1.values, 1: vector2.values}
        children: list[list] = [[], []]

        for i in range(Vector.len()):
            random_num = np.random.randint(low=0, high=2, size=1)  # it is one or zero
            for j in range(2):
                children[j].append(lookup_dict.get((random_num + j) % 2)[i])

        return [Vector(np.array(child)) for child in children]

    @staticmethod
    def __mutate(vector: Vector) -> None:
        values = vector.values
        num_mutations = np.ceil(abs(np.random.normal(size=1)))
        mutation_value = np.random.normal(size=1)
        for i in range(num_mutations):
            rand_int = np.random.randint(low=0, high=Vector.len(), size=1)
            values[rand_int] += mutation_value

        vector.values = values
