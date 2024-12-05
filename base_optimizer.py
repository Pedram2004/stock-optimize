from abc import ABC, abstractmethod
from vector import Vector
import numpy as np

class Optimizer(ABC):
    def __init__(self, num_iterations: int, random_state: int = 42):
        self._num_iterations = num_iterations
        np.random.seed(random_state)

    @abstractmethod
    def optimize(self) -> Vector:
        pass

    @abstractmethod
    def draw_chart(self) -> None:
        pass