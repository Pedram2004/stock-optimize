import numpy as np
from typing import Iterable

class Vector:
    __covariance_matrix: np.array
    __expected_return: np.array

    def __init__(self, vector: Iterable):
        self.__vector = np.array(vector)
        self.__fitness = self.__fitness_func()

    @property
    def vector(self) -> np.array:
        return np.copy(self.__vector)

    @property
    def fitness(self) -> float:
        return self.__fitness

    @classmethod
    def len(cls) -> int:
        return len(cls.__expected_return)

    @classmethod
    def set_class_variables(cls, expected_return, covariance_matrix):
        cls.__covariance_matrix = np.array(covariance_matrix)
        cls.__expected_return = np.array(expected_return)

    def __fitness_func(self) -> float:
        portfolio_risk = np.sqrt(np.matmul(
            np.matmul(self.__vector, Vector.__covariance_matrix),
            self.__vector)[0])

        portfolio_return = np.matmul(self.__vector, Vector.__expected_return)[0]
        return portfolio_return / portfolio_risk
