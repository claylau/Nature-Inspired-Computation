import random
import numpy as np

from .opeator import roulette_wheel_selection
from .coder import encode, decode


class Individual:
    def __init__(self, dimension, lengths, lower_bounds, up_bounds, 
        code_mode, init_mode="solution", chromosome=None):
        self.dimension = dimension
        self.lower_bounds = lower_bounds
        self.up_bounds = up_bounds
        self.lengths = lengths
        self.code_mode = code_mode
        if init_mode == "solution":
            self.solution = self._init_solution(
                self.dimension, self.lower_bounds, self.up_bounds
                )
            self.chromosome = self._init_chromosome()
        elif init_mode == "chromosome":
            self.chromosome = chromosome
            self.solution = list(range(self.dimension))
            self.update_solution()
        else:
            raise ValueError("Invalid arguments for initializing")

    def _init_solution(self, dimension, lower_bounds, up_bounds):
        """Generate a initialization solution from uniform distribution.
        Args:
            dimension (int): number of variables in one solution.
            lower_bounds (float): the lower_bound of solution.
            up_bounds (float): the up_bound of solution.
        """
        solutions = list(range(dimension))
        for i in range(dimension):
            lower_bound = lower_bounds[i]
            up_bound = up_bounds[i]
            var_i = round(random.random() * (up_bound - lower_bound) + lower_bound, 2)
            solutions[i] = var_i
        return solutions
    
    def _init_chromosome(self):
        """Encode the solution represented by the individual
        Return:
            chromosome (list[string]): each string represented the gene of one variable.
        """
        chromosome = []
        for i in range(self.dimension):
            s = self.solution[i]
            lower = self.lower_bounds[i]
            up = self.up_bounds[i]
            l = self.lengths[i]
            gene = encode(s, lower, up, l, self.code_mode)
            chromosome.append(gene)
        return chromosome
    
    def get_fitness(self, fitness_func):
        """Compute the fitness of a solution given the fitness function.
        Args:
            fitness_func (callable): the fitness function to evaluate a solution.
        """
        self.fitness = fitness_func(self.solution)
        return self.fitness
    
    def mutation(self, prob):
        """Alter one or more gene with a probability.
        Args:
            prob (float): mutation probability.
        """
        if self.code_mode == "binary":
            for i in range(self.dimension):
                gene = list(self.chromosome[i])
                neg_gene = ['0' if i == '1' else '1' for i in gene]
                len_gene = self.lengths[i]
                p = np.random.random(len_gene)
                is_muate = p < prob
                for j in range(len_gene):
                    if is_muate[j]:
                        gene[j] = neg_gene[j]
                self.chromosome[i] = ''.join(gene)
        elif self.code_mode == "real":
            raise ValueError("Sorry real code hasn't been implement.")
        else:
            raise ValueError("Unkonw code mode.")
        return
    
    def crossover(self, individual, prob, num_point):
        """crossover with the other individual to generate two offspring.
        """
        if self.code_mode == "binary":
            child1_chromosome = list(range(self.dimension))
            child2_chromosome = list(range(self.dimension))
            for i in range(self.dimension):
                mom_gene = list(self.chromosome[i])
                papa_gene = list(individual.chromosome[i])
                l = self.lengths[i]
                points = random.sample(range(l-1), num_point+1)
                if num_point % 2 != 0:
                    points[-1] = l - 1 
                for j in range(0, num_point, 2):
                    p_start = points[j]
                    p_end = points[j+1] + 1
                    is_exchange = random.random() < prob
                    if is_exchange:
                        tmp = mom_gene[p_start:p_end]
                        mom_gene[p_start:p_end] = papa_gene[p_start:p_end]
                        papa_gene[p_start:p_end] = tmp
                child1_chromosome[i] = ''.join(mom_gene)
                child2_chromosome[i] = ''.join(papa_gene)
        elif self.code_mode == "real":
            raise ValueError("Sorry real code hasn't been implement.")
        else:
            raise ValueError("Unkonw code mode.")
        child1 = Individual(self.dimension, self.lengths, self.lower_bounds, self.up_bounds,
            self.code_mode, init_mode="chromosome", chromosome=child1_chromosome)
        child2 = Individual(self.dimension, self.lengths, self.lower_bounds, self.up_bounds,
            self.code_mode, init_mode="chromosome", chromosome=child2_chromosome)
        return child1, child2

    def update_solution(self):
        """update the solution represented by the individual
        """
        if self.code_mode == "binary":
            for i in range(self.dimension):
                gene = self.chromosome[i]
                lower = self.lower_bounds[i]
                up = self.up_bounds[i]
                l = self.lengths[i]
                self.solution[i] = decode(gene, lower, up, l, self.code_mode)
        elif self.code_mode == "real":
            raise ValueError("Sorry real code hasn't been implement.")
        else:
            raise ValueError("Unkonw code mode.")
        return
    
    def get_obj_value(self, obj_func):
        """Compute the value of the objective function at this solution.
        """
        return obj_func(self.solution)


class Population:
    def __init__(self, init_mode="size", size=100, individuals=None, **kwargs):
        if init_mode == "size":
            self.size = size
            self.individuals = list(range(self.size))
            for i in range(self.size):
                self.individuals[i] = \
                    Individual(
                        kwargs["dimension"], kwargs["lengths"], 
                        kwargs["lower_bounds"], kwargs["up_bounds"],
                        kwargs["code_mode"], init_mode="solution",
                    )
        elif init_mode == "individual":
            self.individuals = individuals
            self.size = len(individuals)
        else:
            raise ValueError("Invalid arguments for initializing")
    
    def mate_selection(self, fitness_func):
        """Mate selection.
        Args:
            fitness_func (callable): fitness function.
        return:
            mate_index (list): mate index in the population.
        """
        mate_index = roulette_wheel_selection(self.individuals, fitness_func, size=self.size)
        return mate_index

    def mutation(self, prob):
        """Mutation in all individual at a probability.
        """
        for individual in self.individuals:
            individual.mutation(prob)
        return
    
    def crossover(self, mate_index, prob, num_point):
        """Crossover: exchange the right parts of two parents to generate offspring.
        Args:
            mate_index (list): mate index in the population.
            prob (float): the probility of exchange.
            num_point (int): the number of point to cut.
        """
        size = len(mate_index)
        offspring = list(range(size*2))
        for i in range(size):
            mom = self.individuals[mate_index[i]]
            papa = self.individuals[(mate_index[i] + 1)%size]
            child1, child2 = mom.crossover(papa, prob, num_point)
            offspring[i] = child1
            offspring[i+size] = child2
        offspring_pop = Population(init_mode="individual", individuals=offspring)
        return offspring_pop
    
    def environment_selection(self, fitness_func, size=None):
        """Random select new population with size based on their fitness_values.
        Implement as Roulette Wheel Approach: if p_{k-1} < r < p_k, then chose k index.
        Args:
            fitness_func (callable): the fitness function to evaluate a solution.
            size (int): the size of new population.
        """
        if size == None:
            size = self.size
        live_index = roulette_wheel_selection(self.individuals, fitness_func, size)
        live_individuals = list(range(size))
        for i in range(size):
            live_individuals[i] = self.individuals[live_index[i]]
        live_pop = Population(init_mode="individual", individuals=live_individuals)
        death_index = list(set(range(self.size)) - set(live_index))
        death_individuals = list(range(len(death_index)))
        for i in range(len(death_index)):
            death_individuals[i] = self.individuals[death_index[i]]
        death_pop = Population(init_mode="individual", individuals=death_individuals)
        return live_pop, death_pop
    
    def evaluation(self, obj_func, mode="maximize"):
        """Evaluation this population.
        """
        result = {}
        solutons = list(range(self.size))
        obj_func_value = list(range(self.size))
        for i in range(self.size):
            individual = self.individuals[i]
            solutons[i] = individual.solution
            obj_func_value[i] = individual.get_obj_value(obj_func)
        if mode == "maximize":
            best_index = np.argmax(obj_func_value)
            best_solution = self.individuals[best_index].solution
            best_value = obj_func_value[best_index]
        elif mode == "minimize":
            best_index = np.argmin(obj_func_value)
            best_solution = self.individuals[best_index].solution
            best_value = obj_func_value[best_index]
        else:
            raise ValueError("Invaild mode.")
        result["best_x"] = best_solution
        result["best_y"] = best_value
        result["mean_y"] = np.mean(obj_func_value)
        result["std_y"] = np.std(obj_func_value)
        result["x"] = solutons
        result["y"] = obj_func_value
        return result

    def update_solution(self):
        """update solution in place.
        """
        for individual in self.individuals:
            individual.update_solution()
        return

    def elite_selection(self, fitness_func, size):
        """return top-k of the population.
        """
        fitness_values = []
        for individual in self.individuals:
            fitness_values.append(individual.get_fitness(fitness_func))
        fitness_values = np.array(fitness_values)
        top = np.argsort(fitness_values)[::-1]
        elite_individuals = list(range(size))
        for i in range(size):
            elite_individuals[i] = self.individuals[top[i]]
        elite_pop = Population(init_mode="individual", individuals=elite_individuals)
        return elite_pop
    
    def __add__(self, pop):
        """population add operator.
        """
        individuals = self.individuals + pop.individuals
        return Population(init_mode="individual", individuals=individuals)
