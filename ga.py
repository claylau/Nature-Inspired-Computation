__author__ = "claylau"
__doc__ = "General Genetic Algorithm for Function Optimalization."
import math
import random
import numpy as np


def initialize(n, lower_bound, up_bound, constrain=None):
    """Generate a number of initialization solutions from uniform distribution.
    Args:
        n (int): number of initialization solutions.
        lower_bound (float): the lower_bound of solution.
        up_bound (float): the up_bound of solution.
        constrain (callable): a constrain function on solution, defalut None.
    """
    solutions = []
    if constrain is None:
        constrain = lambda x: True
    assert (n > 1), "need more than one solution."
    i = 0
    while i < n:
        s = round(random.random() * (up_bound - lower_bound) + lower_bound, 2)
        if constrain(s):
            solutions.append(s)
            i += 1
    return solutions


def encode(solution, lower_bound, up_bound, length, mode="binary"):
    """Encode a solution with binary or float code.
    Args:
        solution (float or int): a single variable to encode.
        length (int): code length.
        mode (binary or real): encode mode, default binary. 
    """
    if mode == "binary":
        precision = (up_bound - lower_bound) / (2 ** length - 1)
        code_dec = int((solution - lower_bound) / precision)
        code_bin = bin(code_dec)
        code = str(code_bin).lstrip("0b").zfill(length)
    elif mode == 'real':
        raise ValueError("Sorry real code hasn't been implement.")
    else:
        raise ValueError("Unkonw code mode.")
    return code


def decode(code, lower_bound, up_bound, length, mode="binary"):
    """Decode a code to a valid solution.
    Args:
        code (binary or real): a single variable to decode.
        length (int): code length.
        mode (binary or real): code type, default binary. 
    """
    if mode == "binary":
        precision = (up_bound - lower_bound) / (2 ** length - 1)
        code_dec = int(code, 2)
        solution = lower_bound + precision * code_dec
    elif mode == 'real':
        raise ValueError("Sorry real code hasn't been implement.")
    else:
        raise ValueError("Unkonw code mode.")
    return solution


def fitness(solution, obj_func, mode='max'):
    """Compute the fitness of solution given the objective function.
    Args:
        obj_func (callable): the objective function to be optimized.
        mode (max or min): optimization direction.
    """
    if mode == "max":
        fitness_value = obj_func(*solution)
    elif mode == "min":
        fitness_value = -obj_func(*solution)
    else:
        raise ValueError("Unkonw mode for fitness compute.")
    return fitness_value


def environment_selection(population, fitness_values, size):
    """Random select new population with size based on their fitness_values.
    Implement as Roulette Wheel Approach: if p_{k-1} < r < p_k, then chose k index.
    Args:
        population (list): list of solutions.
        fitness_values (list): list of each solution's fitness_value.
        size (int): the size of new population.
    """
    fitness_values = np.array(fitness_values)
    sum_of_fitness = np.sum(fitness_values)
    fitness_values /= sum_of_fitness
    added_probility = np.cumsum(fitness_values)
    new_population = list(range(size))
    for i in range(size):
        r = random.random()
        compare = (r > added_probility).astype(np.int8)
        compare_shift = np.zeros_like(compare, dtype=compare.dtype)
        compare_shift[:-1] = compare[1:]
        compare_shift[-1] = 0
        index = np.nonzero(compare - compare_shift)[0]
        if index.size > 0:
            index = index.item() + 1
        else:
            index = 0
        new_population[i] = population[index]
    random.shuffle(new_population)
    return new_population


def crossover(population, prob, num_point=1):
    """Crossover: exchange the right parts of two parents to generate offspring.
    Args:
        population (list): parent population.
        prob (float): the probility of exchange.
        num_point (int): the number of point to cut.
    """
    size = len(population)
    new_population = list(range(size))
    len_chromosome = len(population[0])
    if size % 2 != 0:
        parents_index = random.sample(range(size - 1), size - 1)
        new_population[-1] = population[-1]
    else:
        parents_index = random.sample(range(size), size)
    for i in range(0, len(parents_index), 2):
        mom = list(population[parents_index[i]])
        papa = list(population[parents_index[i+1]])
        points = random.sample(range(len_chromosome-1), num_point+1)
        if num_point % 2 != 0:
            points[-1] = len_chromosome-1 
        for j in range(0, num_point, 2):
            p_start = points[j]
            p_end = points[j+1] + 1
            is_exchange = random.random() < prob
            if is_exchange:
                tmp = mom[p_start:p_end]
                mom[p_start:p_end] = papa[p_start:p_end]
                papa[p_start:p_end] = tmp
        new_population[i] = ''.join(mom)
        new_population[i+1] = ''.join(papa)
    return new_population


def mutation(chromosome, prob, mode="binary"):
    """Alter one or more gene with a probability.
    Args:
        chromosome (list of coded string): chromosome to be mutated.
        prob (float): mutation probability.
        mode (binary or real): code mode of chromosome.
    """
    if mode == "binary":
        chromosome = list(chromosome)
        neg_chromosome = ['0' if i == '1' else '1' for i in chromosome]
        len_chromosome = len(chromosome)
        p = np.random.random(len_chromosome)
        is_muate = p < prob
        for i in range(len_chromosome):
            if is_muate[i]:
                chromosome[i] = neg_chromosome[i]
        chromosome = ''.join(chromosome)
    elif mode == "real":
        raise ValueError("Sorry real code hasn't been implement.")
    else:
        raise ValueError("Unkonw code mode.")
    return chromosome


def evaluation(solutions, obj_func):
    obj_func_value = list(range(len(solutions)))
    for i in range(len(solutions)):
        individual = solutions[i]
        obj_func_value[i] = obj_func(*individual)
    return obj_func_value


def low_dimension_single_objective_optimization():
    cfg = {}
    cfg["generation"] = 100
    cfg["dimension"] = 2
    cfg["obj"] = lambda x1, x2: 1 / (x1**2 + x2**2)
    cfg["lower_bound"] = -5
    cfg["up_bound"] = 5 
    cfg["population_size"] = 20
    cfg["length"] = 8
    cfg["num_point"] = 2
    cfg["pc"] = 0.8
    cfg["pm"] = 0.05
    return cfg


def high_dimension_single_objective_optimization():
    cfg = {}
    cfg["generation"] = 100
    cfg["dimension"] = 5
    cfg["obj"] = lambda x1, x2, x3, x4, x5:\
         1 / (x1**2 + x2**2 + x3**2 + x4**2 + x5**2)
    cfg["lower_bound"] = -5
    cfg["up_bound"] = 5 
    cfg["population_size"] = 40
    cfg["length"] = 8
    cfg["num_point"] = 2
    cfg["pc"] = 0.8
    cfg["pm"] = 0.05
    return cfg

def main():
    # single variable single objective optimization
    cfg = low_dimension_single_objective_optimization()
    # cfg = high_dimension_single_objective_optimization()
    dimension = cfg["dimension"]
    population_size = cfg["population_size"]
    lower_bound = cfg["lower_bound"]
    up_bound = cfg["up_bound"]
    generation = cfg["generation"]
    obj_func = cfg["obj"]
    length = cfg["length"]
    num_point = cfg["num_point"]
    pc = cfg["pc"]
    pm = cfg["pm"]

    tmp_solutions = list(range(dimension))
    for i in range(dimension):
        tmp_solutions[i] = initialize(population_size, lower_bound, up_bound)
    
    solutions = list(range(population_size))
    population = list(range(population_size))
    
    for i in range(population_size):
        x = list(range(dimension))
        chromosome = list(range(dimension))
        for j in range(dimension):
            x[j] = tmp_solutions[j][i]
            chromosome[j] = encode(tmp_solutions[j][i], lower_bound, up_bound, length)
        solutions[i] = x
        population[i] = ''.join(chromosome)
    
    eval_value = evaluation(solutions, obj_func)
    best_index = np.argmax(eval_value)
    print('generation 0: best solution {}, obj_func value {}'.format(
        solutions[best_index], eval_value[best_index]))
    for i in range(1, generation):
        fitness_values = [fitness(s, obj_func) for s in solutions]
        population = environment_selection(population, fitness_values, len(population))
        population = crossover(population, pc, num_point)
        population = [mutation(chromosome, pm) for chromosome in population]
        solutions = list(range(len(population)))
        for p in range(len(population)):
            individual = population[p]
            solution = list(range(dimension))
            for d in range(dimension):
                code = individual[d*length:(d+1)*length]
                solution[d] = decode(code, lower_bound, up_bound, length)
            solutions[p] = solution
        eval_value = evaluation(solutions, obj_func)
        best_index = np.argmax(eval_value)
        print('generation {}: best solution {}, obj_func value {}'.format(
            i, solutions[best_index], eval_value[best_index]))


if __name__ == "__main__":
    main()