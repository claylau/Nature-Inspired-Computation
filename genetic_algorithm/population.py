import random

import numpy as np

from .coder import decode, encode
from .individual import Individual
from .operator import roulette_wheel_selection


class Population:
    def __init__(self, size, individuals=None, **kwargs):
        self.size = size
        if individuals == None:
            self.individuals = list(range(self.size))
            for i in range(self.size):
                self.individuals[i] = Individual(
                    dimension=kwargs["dimension"],
                    lower_bounds=kwargs["lower_bounds"],
                    up_bounds=kwargs["up_bounds"],
                    lengths=kwargs["lengths"],
                    code_mode=kwargs["code_mode"]
                )
        else:
            self.individuals = individuals
            assert self.size == len(individuals)
    
    # def mate_selection(self, fitness_func):
    #     """Mate selection.
    #     Args:
    #         fitness_func (callable): fitness function.
    #     return:
    #         mate_index (list): mate index in the population.
    #     """
    #     mate_index = roulette_wheel_selection(self.individuals, fitness_func, size=self.size)
    #     return mate_index

    def mutation(self, prob, index=None):
        """Mutation in all individual at a probability.
        """
        for individual in self.individuals:
            individual.mutation(prob, index)
        return
    
    def crossover(self, mate_index, prob, cross_point=None, index=None):
        """Crossover: exchange the right parts of two parents to generate offspring.
        Args:
            mate_index (list): mate index in the population.
            prob (float): the probility of exchange.
            cross_point (int): the number of point to cut.
        """
        size = len(mate_index)
        offspring = list(range(size*2))
        for i in range(size):
            mom = self.individuals[mate_index[i]]
            papa = self.individuals[(mate_index[i] + 1)%size]
            child1, child2 = mom.crossover(papa, prob, cross_point, index)
            offspring[i] = child1
            offspring[i+size] = child2
        offspring_pop = Population(size=size*2, individuals=offspring)
        return offspring_pop
    
    # def environment_selection(self, fitness_func, size=None):
    #     """Random select new population with size based on their fitness_values.
    #     Implement as Roulette Wheel Approach: if p_{k-1} < r < p_k, then chose k index.
    #     Args:
    #         fitness_func (callable): the fitness function to evaluate a solution.
    #         size (int): the size of new population.
    #     """
    #     if size == None:
    #         size = self.size
    #     live_index = roulette_wheel_selection(self.individuals, fitness_func, size)
    #     live_individuals = list(range(size))
    #     for i in range(size):
    #         live_individuals[i] = self.individuals[live_index[i]]
    #     live_pop = Population(init_mode="individual", individuals=live_individuals)
    #     death_index = list(set(range(self.size)) - set(live_index))
    #     death_individuals = list(range(len(death_index)))
    #     for i in range(len(death_index)):
    #         death_individuals[i] = self.individuals[death_index[i]]
    #     death_pop = Population(init_mode="individual", individuals=death_individuals)
    #     return live_pop, death_pop
    
    # def evaluation(self, obj_func, mode="maximize"):
    #     """Evaluation this population.
    #     """
    #     #TODO implement multi-object optimal
    #     result = {}
    #     solutons = list(range(self.size))
    #     obj_func_value = list(range(self.size))
    #     for i in range(self.size):
    #         individual = self.individuals[i]
    #         solutons[i] = individual.solution
    #         obj_func_value[i] = individual.get_obj_value(obj_func)
    #     if mode == "maximize":
    #         best_index = np.argmax(obj_func_value)
    #         best_solution = self.individuals[best_index].solution
    #         best_value = obj_func_value[best_index]
    #     elif mode == "minimize":
    #         best_index = np.argmin(obj_func_value)
    #         best_solution = self.individuals[best_index].solution
    #         best_value = obj_func_value[best_index]
    #     else:
    #         raise ValueError("Invaild mode.")
    #     result["best_x"] = best_solution
    #     result["best_y"] = best_value
    #     result["mean_y"] = np.mean(obj_func_value)
    #     result["std_y"] = np.std(obj_func_value)
    #     result["x"] = solutons
    #     result["y"] = obj_func_value
    #     return result

    def update_solution(self):
        """update solution in place.
        """
        for individual in self.individuals:
            individual.update_solution()
        return

    # def elite_selection(self, fitness_func, size):
    #     """return top-k of the population.
    #     """
    #     fitness_values = []
    #     for individual in self.individuals:
    #         fitness_values.append(individual.get_fitness(fitness_func))
    #     fitness_values = np.array(fitness_values)
    #     top = np.argsort(fitness_values)[::-1]
    #     elite_individuals = list(range(size))
    #     for i in range(size):
    #         elite_individuals[i] = self.individuals[top[i]]
    #     elite_pop = Population(init_mode="individual", individuals=elite_individuals)
    #     return elite_pop
    
    def __add__(self, pop):
        """population add operator.
        """
        individuals = self.individuals + pop.individuals
        size = self.size + pop.size
        return Population(size=size, individuals=individuals)
