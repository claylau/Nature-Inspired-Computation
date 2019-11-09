import random
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
    fitness_values = np.array(fitness_values) - np.min(fitness_values)
    sum_of_fitness = np.sum(fitness_values)
    if np.absolute(sum_of_fitness) < 1e-5:
        index = list(range(size))
    else:
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


def tournament_selection(individuals, comparison_operator, size, part=2):
    """Tournament Selection.
    Args:
        individuals (list[Individual]): list of individual.
        comparsion_operator (callable): compare two participants.
    Return:
        index (list): index of original pop.
    """
    index = list(range(size))
    for i in range(size):
        participants = random.sample(range(len(individuals)), part)
        best = None
        for p in participants:
            if best == None or comparison_operator(individuals[p], individuals[best]) == 1:
                best = p
        index[i] = best
    return index


def fast_nondominated_sort(obj_values, optimal_direction):
    """Fast Nondominated Sort proposed by NSGA-II.
    Args:
        obj_values (list[list[obj_value]]): 
            list of each individual's obj_func values in list style.
        optimal_direction (list["maximize" or "minimize"]): 
            object optimal direction used to judge domination.
    Return:
        ranks (list): front ids of each individuals.
        fronts (list[set]): 
            list of each level front, each element is a set include the index of individuals. 
    """
    def is_dominate(p, q, optimal_direction):
        """Return (int):
            1: p dominate q,
            -1: q dominate p,
            0: p nodominate q.
        """
        l = len(p)
        sum_l = l
        p_better = list(range(l))
        for i in range(l):
            pv, qv, od = p[i], q[i], optimal_direction[i]
            if ((pv - qv) > 0 and od == "maximize") or \
                ((pv - qv) < 0 and od == "minimize"):
                p_better[i] = 1
            elif ((pv - qv) > 0 and od == "minimize") or \
                ((pv - qv) < 0 and od == "maximize"):
                p_better[i] = -1
            else:
                p_better[i] = 0
                sum_l -= 1
        if sum(p_better) == sum_l:
            return 1
        elif sum(p_better) == -sum_l:
            return -1
        else:
            return 0
    
    size = len(obj_values)
    ranks = list(range(size))
    fronts = [set() for i in range(size)]
    # a set of solutions that the solution p dominates.
    p_dominate = [set() for i in range(size)]
    # the number of solutions which dominate the solution p.
    dominate_p = [0 for i in range(size)]
    for i in range(size):
        for j in range(size):
            r = is_dominate(obj_values[i], obj_values[j], optimal_direction)
            if r == 1:
                p_dominate[i].add(j)
            elif r == -1:
                dominate_p[i] += 1
        if dominate_p[i] == 0:
            ranks[i] = 0
            fronts[0].add(i)
    i = 0
    while len(fronts[i]) > 0:
        for j in fronts[i]:
            for k in p_dominate[j]:
                dominate_p[k] -= 1
                if dominate_p[k] == 0:
                    ranks[k] = i + 1
                    fronts[i+1].add(k)
        i += 1
    fronts = [list(f) for f in fronts if len(f)]
    return ranks, fronts


def crowding_distance_assignment(obj_values):
    """Crowding Distance Assignment in the front proposed by NSGA-II.
    Args:
        obj_values (list[list[obj_value]]): 
            list of each individual's obj_func values in list style.
    Return:
        distances (list): 
            each individual crowding distance in the front.
    """
    front_nums = len(obj_values)
    distances = [0 for i in range(front_nums)]
    obj_func_nums = len(obj_values[0])
    for m in range(obj_func_nums):
        sorted_index = sorted(range(len(obj_values)), key=lambda x: obj_values[x][m])
        distances[sorted_index[0]] = float("inf")
        distances[sorted_index[-1]] = float("inf") 
        f = obj_values[sorted_index[-1]][m] - obj_values[sorted_index[0]][m]
        for i in range(1, front_nums-1):
            j_last = sorted_index[i-1]
            j = sorted_index[i]
            j_next = sorted_index[i+1]
            distances[j] += (obj_values[j_next][m] - obj_values[j_last][m]) / (f+1e-4)
    return distances


def crowd_comparison(p, q):
    """Crowded Comparison Operator. Return 1 means p is better than q.
    Return -1 means q is better than p.
    """
    if p.rank < q.rank or (p.rank == q.rank and p.distance > q.distance):
        return 1
    elif p.rank > q.rank or (p.rank == q.rank and p.distance < q.distance):
        return -1
    else:
        return 0