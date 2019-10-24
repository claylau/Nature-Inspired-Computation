import numpy as np

def roulette_wheel_selection(individuals, fitness_func, size):
    """Roulette Wheel Selection Rule.
    Args:
        individuals (list[Individual]): list of individual.
        fitness_func (callable): fitness function.
        size (int): size of return. Default `len(pop)`
    Return:
        index (list): index of original pop.
    """
    fitness_values = []
    for individual in individuals:
        fitness_values.append(individual.get_fitness(fitness_func))
    fitness_values = np.array(fitness_values)
    sum_of_fitness = np.sum(fitness_values)
    fitness_values /= sum_of_fitness
    added_probility = np.cumsum(fitness_values)
    index = list(range(size))
    for i in range(size):
        r = np.random.random()
        compare = (r > added_probility).astype(np.int8)
        compare_shift = np.zeros_like(compare, dtype=compare.dtype)
        compare_shift[:-1] = compare[1:]
        compare_shift[-1] = 0
        ind = np.nonzero(compare - compare_shift)[0]
        if ind.size == 1:
            ind = ind.item() + 1
        elif ind.size > 1:
            ind = ind[0].item() + 1
        else:
            ind = 0
        index[i] = ind
    np.random.shuffle(index)
    return index