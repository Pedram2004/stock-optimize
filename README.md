# Stock Optimize

## Introduction

**Stock Optimize** is a Python project that implements various optimization algorithms to solve the portfolio optimization problem. The goal is to determine the optimal allocation of assets in a stock portfolio to maximize expected returns while minimizing risk.

This project was completed as part of the **Artificial Intelligence** course taken in the Fall of 2024 at Ferdowsi University of Mashhad.

## Algorithms Implemented

The project includes the following optimization algorithms:

- **Genetic Algorithm**
- **Beam Search**
- **Random Beam Search**
- **Simulated Annealing**

These algorithms are defined in the `Optimizer` module and can be utilized to find the optimal investment strategy.

## Installation

Ensure you have Python 3 installed. Install the required packages using:

```
pip install numpy matplotlib
```

## Project Background

This project was completed as part of the **Artificial Intelligence** course in the Fall of 2024. It applies various optimization algorithms to solve the portfolio optimization problem, focusing on maximizing expected returns while minimizing risk.

## Classes and Modules Description

### **Optimizer Module**

The `Optimizer` module contains implementations of different optimization algorithms.

#### **Vector Class**

Represents an investment vector in the optimization process.

- **Attributes:**
  - `__covariance_matrix`: Covariance matrix of asset returns.
  - `__expected_return`: Expected returns for each asset.
  - `__max_comparison`: Boolean indicating comparison mode for fitness.
  - `__number_children`: Number of children vectors in certain algorithms.
  - `__values`: Allocation proportions for each asset.
  - `__fitness`: Fitness value of the vector.

- **Methods:**
  - `__init__(self, vector)`: Initializes the vector with allocation values and calculates fitness.
  - `__eq__(self, other)`: Checks equality between two vectors.
  - `__lt__(self, other)`: Defines less-than comparison based on fitness.
  - `values(self)`: Getter and setter for `__values`, ensures allocations sum to 1.
  - `fitness(self)`: Returns the fitness value.
  - `len(cls)`: Class method to get the number of assets.
  - `max_comparison(cls, boolean_value)`: Sets comparison mode for fitness evaluation.
  - `set_class_variables(cls, expected_return, covariance_matrix)`: Sets class-level variables.
  - `__fitness_func(self)`: Calculates the fitness as return-to-risk ratio.
  - `get_neighbours(self, learning_rate)`: Generates neighbouring vectors for exploration.

#### **Optimizer Base Class**

An abstract base class defining the structure for optimization algorithms.

- **Attributes:**
  - `_num_iterations`: Number of iterations for the optimizer.

- **Methods:**
  - `optimize(self)`: Abstract method to perform optimization.
  - `get_plot_info(self)`: Abstract method to retrieve data for plotting.

### **Optimization Algorithms**

#### **GeneticAlgorithmOptimizer Class**

Implements the Genetic Algorithm for optimization.

- **Attributes:**
  - `__num_individuals`: Population size.
  - `__population`: List of current vectors in the population.
  - `__iterations`: Fitness values over iterations.
  - `__mutation_normal_deviation`: Standard deviation for mutation scaling.

- **Methods:**
  - `__init__(self, num_individuals, num_iterations, random_state)`: Initializes the optimizer.
  - `optimize(self)`: Executes the genetic algorithm steps: selection, crossover, and mutation.
  - `get_plot_info(self)`: Provides data for plotting convergence.

#### **BeamSearchOptimizer Class**

Implements Beam Search optimization.

- **Attributes:**
  - `__beam_length`: Number of top vectors to retain each iteration.
  - `__beam`: Current list of vectors in the beam.
  - `__LEARNING_RATE`: Step size for generating neighbours.
  - `__iterations`: Fitness values over iterations.

- **Methods:**
  - `__init__(self, beam_length, num_iterations, learning_rate, random_state)`: Initializes the optimizer.
  - `optimize(self)`: Iteratively expands and selects the best neighbours.
  - `get_plot_info(self)`: Provides data for plotting convergence.

#### **RandomBeamSearchOptimizer Class**

Implements Random Beam Search optimization, a variation of Beam Search.

- **Attributes and Methods:**

  Similar to `BeamSearchOptimizer`, but selection of beams is based on probabilistic sampling rather than deterministic ranking.

#### **SimulatedAnnealingOptimizer Class**

Implements Simulated Annealing optimization.

- **Attributes:**
  - `__current_state`: Current solution vector.
  - `__LEARNING_RATE`: Step size for neighbour generation.
  - `__neighbours`: Tuple containing the current state and its neighbours.
  - `__iterations`: Fitness values over iterations.

- **Methods:**
  - `__init__(self, num_iterations, learning_rate, random_state)`: Initializes the optimizer.
  - `optimize(self)`: Performs optimization using a temperature schedule to accept or reject new states.
  - `__perturbate(self)`: Generates a new candidate solution.
  - `__accept_change(self, new_state, temperature)`: Determines acceptance of new state based on probability.
  - `get_plot_info(self)`: Provides data for plotting convergence.

### **Main Module**

The `main.py` script coordinates the execution of the optimization algorithms.

- **Functions:**
  - `combine_plots(optimizers)`: Plots the convergence of each optimizer for comparison.

- **Workflow:**
  - Reads input data from `input.txt` (expected returns and covariance matrix).
  - Sets class variables for the `Vector` class.
  - Initializes each optimizer with specified parameters.
  - Runs the optimization algorithms and outputs the optimal allocations.
  - Visualizes the performance of each algorithm using plots.
