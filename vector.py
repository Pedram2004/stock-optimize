import numpy as np


class Vector:
    __covariance_matrix: np.array
    __expected_return: np.array

    def __init__(self, vector: np.array):
        self.__values = np.divide(vector, np.sum(vector))
        self.__fitness = self.__fitness_func()

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.__values == other.__values

        return False

    def __lt__(self, other):
        if isinstance(other, Vector):
            return self.__fitness < other.__fitness

        return False

    @property
    def values(self) -> np.array:
        return np.copy(self.__values)

    @values.setter
    def values(self, vector: np.array):
        self.__values = np.divide(vector, np.sum(vector))

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
            np.matmul(self.__values, Vector.__covariance_matrix),
            self.__values)[0])

        portfolio_return = np.matmul(self.__values, Vector.__expected_return)[0]
        return portfolio_return / portfolio_risk
