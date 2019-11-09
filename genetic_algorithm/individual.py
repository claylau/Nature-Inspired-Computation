import random
import numpy as np

from .coder import encode, decode


class Individual:
    def __init__(self, dimension, lower_bounds, up_bounds,
        lengths, code_mode, chromosome=None):
        self.dimension = dimension
        self.lower_bounds = lower_bounds
        self.up_bounds = up_bounds
        self.lengths = lengths
        self.code_mode = code_mode
        if chromosome == None:
            self.solution = self._init_solution(
                self.dimension, self.lower_bounds, self.up_bounds
                )
            self.chromosome = self._init_chromosome()
        else:
            self.chromosome = chromosome
            self.solution = list(range(self.dimension))
            self.update_solution()

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
            var_i = random.random() * (up_bound - lower_bound) + lower_bound
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
    
    def mutation(self, prob, index=None):
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
            # polynomial mutation
            for i in range(self.dimension):
                if random.random() > prob:
                    continue
                u = random.random()
                if u <= 0.5:
                    q = (2*u+(1-2*u)*(1-(self.chromosome[i]-self.lower_bounds[i])\
                        /(self.up_bounds[i]-self.lower_bounds[i]))**(index+1))**(1/(index+1))-1
                else:
                    q = 1-(2*(1-u)+2*(u-0.5)*(1-(self.up_bounds[i]-self.chromosome[i])\
                        /(self.up_bounds[i]-self.lower_bounds[i]))**(index+1))**(1/(index+1))
                self.chromosome[i] += q*(self.up_bounds[i]-self.lower_bounds[i])
                if self.chromosome[i] > self.up_bounds[i] or self.chromosome[i] < self.lower_bounds[i]:
                    self.chromosome[i] = self.lower_bounds[i] if random.random() > 0.5 else self.up_bounds[i]
        else:
            raise ValueError("Unkonw code mode.")
        return
    
    def crossover(self, individual, prob, cross_point=None, index=None):
        """crossover with the other individual to generate two offspring.
        """
        if self.code_mode == "binary":
            child1_chromosome = list(range(self.dimension))
            child2_chromosome = list(range(self.dimension))
            for i in range(self.dimension):
                mom_gene = list(self.chromosome[i])
                papa_gene = list(individual.chromosome[i])
                l = self.lengths[i]
                points = random.sample(range(l-1), cross_point+1)
                if cross_point % 2 != 0:
                    points[-1] = l - 1 
                for j in range(0, cross_point, 2):
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
            # SBX
            child1_chromosome = list(range(self.dimension))
            child2_chromosome = list(range(self.dimension))
            for i in range(self.dimension):
                mom_gene = self.chromosome[i]
                papa_gene = individual.chromosome[i]
                u = random.random()
                p1, p2 = sorted([mom_gene, papa_gene])
                if p2 == p1 or random.random() > prob:
                    child1_chromosome[i] = child2_chromosome[i] = p1
                    continue
                a1 = 2 - (1+2*(p1-self.lower_bounds[i])/(p2-p1))**(-index-1)
                a2 = 2 - (1+2*(self.up_bounds[i]-p2)/(p2-p1))**(-index-1)
                if u <= 1/a1:
                    o1 = 0.5*((p1+p2) - (u*a1)**(1/(index+1))*(p2-p1))
                else:
                    o1 = 0.5*((p1+p2) - (1/(2-u*a1))**(1/(index+1))*(p2-p1))
                if u <= 1/a2:
                    o2 = 0.5*((p1+p2) + (u*a2)**(1/(index+1))*(p2-p1))
                else:
                    o2 = 0.5*((p1+p2) + (1/(2-u*a2))**(1/(index+1))*(p2-p1))
                child1_chromosome[i] = o1
                child2_chromosome[i] = o2
        else:
            raise ValueError("Unkonw code mode.")
        child1 = Individual(self.dimension, self.lower_bounds, self.up_bounds,
            self.lengths, self.code_mode, chromosome=child1_chromosome)
        child2 = Individual(self.dimension, self.lower_bounds, self.up_bounds,
            self.lengths, self.code_mode, chromosome=child2_chromosome)
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
            for i in range(self.dimension):
                self.solution[i] = self.chromosome[i]
        else:
            raise ValueError("Unkonw code mode.")
        return
    
    def get_obj_values(self, obj_funcs):
        """Compute the value of each objective function at this solution.
        Args:
            obj_funcs (list[callable]): list of obj_func.
        Return:
            func_values (list[callable]): list of function values.
        """
        func_values = []
        for obj_func in obj_funcs:
            func_values.append(obj_func(self.solution))
        return func_values
    
    def set_fitness(self, fitness):
        """Set Fitness Value to this individual.
        """
        self.fitness = fitness
        return
    
    def set_nondomination_rank(self, rank):
        """Set Nondomination Rank to this individual.
        Args:
            rank (int): the index of front which belong to.
        """
        self.rank = rank
        return
    
    def set_crowding_distance(self, distance):
        """Set Crowding Distance to this individual.
        Args:
            distance (float): the distance that in one front set.
        """
        self.distance = distance
        return