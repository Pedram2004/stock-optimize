from vector import Vector
from optimizer import Optimizer, GeneticAlgorithmOptimizer, BeamSearchOptimizer, RandomBeamSearchOptimizer, SimulatedAnnealingOptimizer
from matplotlib import pyplot as plt

def combine_plots(optimizers: list[Optimizer]):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for optimizer, ax in zip(optimizers, axs.flat):
        chart = optimizer.get_plot()
        for line in chart.axes[0].get_lines():
            ax.plot(line.get_xdata(), line.get_ydata())
        ax.set_title(chart.axes[0].get_title())
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    with open('input.txt', 'r') as file:
        lines = file.readlines()
        expected_return = list(map(float, lines[0].split()))
        covariance_matrix = [list(map(float, line.split())) for line in lines[1:]]
    
    Vector.set_class_variables(expected_return, covariance_matrix)

    print("-----------------Beam Search-----------------")
    beam_search = BeamSearchOptimizer(10, 100, 0.001)
    answer = beam_search.optimize()
    print("fitness: ", answer.fitness)
    print("values: ", answer.values)

    print("-----------------Random Beam Search-----------------")
    random_beam_search = RandomBeamSearchOptimizer(10, 100, 0.001)
    answer = random_beam_search.optimize()
    print("fitness: ", answer.fitness)
    print("values: ", answer.values)

    print("-----------------Simulated Annealing-----------------")
    simulated_annealing = SimulatedAnnealingOptimizer(100, 0.001)
    answer = simulated_annealing.optimize()
    print("fitness: ", answer.fitness)
    print("values: ", answer.values)

    print("-----------------Genetic Algorithm-----------------")
    genetic = GeneticAlgorithmOptimizer(100, 100)
    answer = genetic.optimize()
    print("fitness: ", answer.fitness)
    print("values: ", answer.values)

    combine_plots([beam_search, random_beam_search, simulated_annealing, genetic])  
    