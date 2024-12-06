from vector import Vector
from optimizer import GeneticAlgorithmOptimizer, BeamSearchOptimizer, RandomBeamSearchOptimizer, SimulatedAnnealingOptimizer

if __name__ == '__main__':
    with open('input.txt', 'r') as file:
        lines = file.readlines()
        expected_return = list(map(float, lines[0].split()))
        covariance_matrix = [list(map(float, line.split())) for line in lines[1:]]
    
    Vector.set_class_variables(expected_return, covariance_matrix)

    print("-----------------Beam Search-----------------")
    beam_search = BeamSearchOptimizer(4, 100)
    answer = beam_search.optimize()
    print("fitness: ", answer.fitness)
    print("values: ", answer.values)

    print("-----------------Random Beam Search-----------------")
    random_beam_search = RandomBeamSearchOptimizer(4, 100)
    answer = random_beam_search.optimize()
    print("fitness: ", answer.fitness)
    print("values: ", answer.values)

    print("-----------------Simulated Annealing-----------------")
    simulated_annealing = SimulatedAnnealingOptimizer(100)
    answer = simulated_annealing.optimize()
    print("fitness: ", answer.fitness)
    print("values: ", answer.values)

    print("-----------------Genetic Algorithm-----------------")
    genetic = GeneticAlgorithmOptimizer(100, 100)
    answer = genetic.optimize()
    print(answer)
    