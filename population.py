from vector import Vector
import numpy as np


class Population:
    def __init__(self, num_individuals: int, num_iteration: int):
        self.__num_iteration = num_iteration
        self.__population: list[Vector] = [Population.__creating_individual() for _ in range(num_individuals)]

    @staticmethod
    def __creating_individual() -> Vector:
        random_vector = np.random.uniform(size=Vector.len())
        adjusted_vector = np.divide(random_vector, np.sum(random_vector))
        return Vector(adjusted_vector)

