from Optimizer.vector import Vector
from Optimizer.optimizer import Optimizer, GeneticAlgorithmOptimizer, BeamSearchOptimizer, RandomBeamSearchOptimizer, SimulatedAnnealingOptimizer
import matplotlib.pyplot as plt
from time import time

def combine_plots(optimizers: dict[str, Optimizer]):
    col = {
        'Beam Search': 'blue',
        'Random Beam Search': 'orange',
        'Simulated Annealing': 'green',
        'Genetic Algorithm': 'red'
    }
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for ax, (optimizer_name, optimizer) in zip(axs, optimizers.items()):
        x, y = optimizer.get_plot_info()
        ax.plot(x, y, color=col[optimizer_name])
        ax.set_title(f'{optimizer_name} - Solution Fitness: {y[-1]}')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fitness')
        ax.legend([optimizer_name])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    with open('input.txt', 'r') as file:
        lines = file.readlines()
        expected_return = list(map(float, lines[0].split()))
        covariance_matrix = [list(map(float, line.split())) for line in lines[1:]]
    
    Vector.set_class_variables(expected_return, covariance_matrix)
    
    # random_state = int(time())
    random_state = 42

    print("-----------------Beam Search-----------------")
    beam_search = BeamSearchOptimizer(20, 100, 0.5, random_state)
    answer = beam_search.optimize()
    print("fitness: ", answer.fitness)
    print("values: ", answer.values)

    print("-----------------Random Beam Search-----------------")
    random_beam_search = RandomBeamSearchOptimizer(50, 200, 0.15, random_state)
    answer = random_beam_search.optimize()
    print("fitness: ", answer.fitness)
    print("values: ", answer.values)

    print("-----------------Simulated Annealing-----------------")
    simulated_annealing = SimulatedAnnealingOptimizer(10000, 0.5, random_state)
    answer = simulated_annealing.optimize()
    print("fitness: ", answer.fitness)
    print("values: ", answer.values)

    print("-----------------Genetic Algorithm-----------------")
    genetic = GeneticAlgorithmOptimizer(200, 1000, random_state)
    answer = genetic.optimize()
    print("fitness: ", answer.fitness)
    print("values: ", answer.values)

    combine_plots({
        'Beam Search': beam_search,
        'Random Beam Search': random_beam_search,
        'Simulated Annealing': simulated_annealing,
        'Genetic Algorithm': genetic
    })