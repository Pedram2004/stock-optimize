import numpy as np


class Vector:
    __covariance_matrix: np.array
    __expected_return: np.array
    __max_comparison: bool = False
    __number_children: int

    def __init__(self, vector: np.array):
        self.values = vector
        self.__fitness = self.__fitness_func()

    def __eq__(self, other):
        if isinstance(other, Vector):
            return (self.__values == other.__values).all()

        return False

    def __lt__(self, other):
        if isinstance(other, Vector):
            boolean_value = self.__fitness < other.__fitness
            if Vector.__max_comparison:
                boolean_value = not boolean_value

            return boolean_value

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
    def max_comparison(cls, boolean_value: bool):
        cls.__max_comparison = boolean_value

    @classmethod
    def set_class_variables(cls, expected_return: list, covariance_matrix: list[list]):
        cls.__covariance_matrix = np.array(covariance_matrix)
        cls.__expected_return = np.array(expected_return)
        cls.__number_children = 2 * cls.len() if cls.len() > 5 else 10

    def __fitness_func(self) -> float:
        portfolio_risk = np.sqrt(np.matmul(
            np.matmul(self.__values, Vector.__covariance_matrix),
            self.__values))

        portfolio_return = np.matmul(self.__values, Vector.__expected_return)
        return portfolio_return / portfolio_risk

    @staticmethod
    def __vector_projection(vector: np.array) -> np.array:
        plane_normal_vector = np.ones(Vector.len())
        plane_normal_vector_length = np.linalg.norm(plane_normal_vector)
        proj_i = np.subtract(vector, plane_normal_vector * (np.dot(vector, plane_normal_vector)
                                                            / (plane_normal_vector_length ** 2)))
        return proj_i / np.linalg.norm(proj_i)

    # def get_neighbours(self, radius: float) -> list["Vector"]:
    #     unit_vectors = np.eye(Vector.len())
    #     projected_vectors = []
    #     for unit_vector in unit_vectors:
    #         projected_vector = np.add(self.__values, Vector.__vector_projection(unit_vector) * radius)
    #         if (abs(1 - projected_vector.sum()) <= 10 ** -5) and min(projected_vector) >= 0:
    #             projected_vectors.append(projected_vector)

    #     num_deficient_vectors = int(len(projected_vectors) - (Vector.__number_children / 2))
    #     if num_deficient_vectors < 0:
    #         parent_vectors = projected_vectors.copy()
    #         for i in range(abs(num_deficient_vectors)):
        #        if len(parent_vectors) >= 2:
        #             parent1_index = np.random.randint(low=0, high=len(parent_vectors))
        #             parent2_index = (parent1_index + 1) % len(parent_vectors)
        #             child_vector = parent_vectors[parent1_index] + parent_vectors[parent2_index]
        #             del parent_vectors[parent1_index:parent2_index + 1]
        #             projected_vectors.append(child_vector)
    #     elif num_deficient_vectors > 0:
    #         for i in range(num_deficient_vectors):
    #             vector_index = np.random.randint(low=0, high=len(projected_vectors))
    #             projected_vectors.pop(vector_index)

    #     projected_vectors.extend([-1 * neighbour_vector for neighbour_vector in projected_vectors])
    #     return [Vector(neighbour_vector) for neighbour_vector in projected_vectors]

    def get_neighbours(self, learning_rate: float) -> list["Vector"]:
        neighbours = []
        for index in range(Vector.len()):
            mutation_value = np.random.normal(scale=learning_rate)
            for i in [-1, 1]:
                values = self.values.copy()
                values[index] += mutation_value * i
                values = np.clip(values, 0, 1)
                neighbours.append(Vector(values))
        return neighbours