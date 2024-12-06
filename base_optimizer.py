from abc import ABC, abstractmethod
from vector import Vector
import numpy as np
import matplotlib.pyplot as plt

class Optimizer(ABC):
    def __init__(self, num_iterations: int, random_state: int):
        self._num_iterations = num_iterations
        np.random.seed(random_state)

    @abstractmethod
    def optimize(self) -> Vector:
        pass

    @abstractmethod
    def get_plot(self) -> plt.Figure:
        pass