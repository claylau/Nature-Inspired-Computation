import time

import matplotlib.pyplot as plt
import numpy as np

from .operator import (crowd_comparison, crowding_distance_assignment,
                       fast_nondominated_sort, tournament_selection)
from .population import Population


class NSGA:
    def __init__(self, cfg, problem):
        self.cfg = cfg
        self.problem = problem
        if isinstance(self.cfg.POPULATION.LENGTHS, int):
            self.cfg.POPULATION.LENGTHS = [self.cfg.POPULATION.LENGTHS] * self.problem.dimension
        else:
            assert isinstance(self.cfg.POPULATION.LENGTHS, (list, tuple))
    
    def run(self, save_dir="."):
        pop = Population(
            size=self.cfg.POPULATION.POP_SIZE, dimension=self.problem.dimension,
            lower_bounds=self.problem.lower_bounds, up_bounds=self.problem.up_bounds,
            lengths=self.cfg.POPULATION.LENGTHS, code_mode=self.cfg.POPULATION.CODE_MODE
        )
        logger = list(range(self.cfg.POPULATION.GENERATION))
        logger_file = open(f"{save_dir}/{self.problem.name}.log", "w")
        print(self.cfg, file=logger_file)
        
        obj_values = [individual.get_obj_values(self.problem.obj_funcs) for individual in pop.individuals]
        ranks, fronts = fast_nondominated_sort(obj_values, self.problem.optimal_direction)
        for front in fronts:
            values = [obj_values[i] for i in front]
            distances = crowding_distance_assignment(values)
            for i, d in zip(front, distances):
                pop.individuals[i].set_crowding_distance(d)
        for i in range(pop.size):
            pop.individuals[i].set_nondomination_rank(ranks[i])
        self.plot_front(pop, fig_name='./0')
        logger[0] = self.evaluate_igd(pop)
        since = time.time()
        for i in range(1, self.cfg.POPULATION.GENERATION):
            # if i % 20 == 0:
            #     self.plot_front(pop, fig_name=f'./{i}')
            # generate offspring
            mate_index = tournament_selection(pop.individuals, crowd_comparison, pop.size)
            offspring_pop = pop.crossover(mate_index, self.cfg.POPULATION.PC, 
                self.cfg.POPULATION.CROSS_POINTS, self.cfg.POPULATION.CINDEX)
            offspring_pop.mutation(self.cfg.POPULATION.PM, self.cfg.POPULATION.MINDEX)
            offspring_pop.update_solution()
            pop = offspring_pop + pop
            # environment selection, generate next generation.
            obj_values = [individual.get_obj_values(self.problem.obj_funcs) for individual in pop.individuals]
            ranks, fronts = fast_nondominated_sort(obj_values, self.problem.optimal_direction)
            sum_count = 0
            next_index = []
            for f in range(len(fronts)):
                sum_count += len(fronts[f])
                if sum_count <= self.cfg.POPULATION.POP_SIZE:
                    front = fronts[f]
                    values = [obj_values[j] for j in front]
                    distances = crowding_distance_assignment(values)
                    for j, d in zip(front, distances):
                        pop.individuals[j].set_crowding_distance(d)
                        pop.individuals[j].set_nondomination_rank(f)
                    next_index += front
                else:
                    break
            if sum_count - len(fronts[f]) < self.cfg.POPULATION.POP_SIZE:
                front = fronts[f]
                # crowding distance sort
                values = [obj_values[j] for j in front]
                distances = crowding_distance_assignment(values)
                for j, d in zip(front, distances):
                    pop.individuals[j].set_crowding_distance(d)
                    pop.individuals[j].set_nondomination_rank(f)
                first_index = sorted(range(len(distances)), key=lambda x: distances[x], reverse=True)
                next_index += [front[first_index[j]] for j in range(self.cfg.POPULATION.POP_SIZE - len(next_index))]
            assert len(next_index) == self.cfg.POPULATION.POP_SIZE
            new_individuals = list(range(len(next_index)))
            for j in range(len(next_index)):
                new_individuals[j] = pop.individuals[next_index[j]]
            pop = Population(size=self.cfg.POPULATION.POP_SIZE, individuals=new_individuals)
            igd = self.evaluate_igd(pop)
            print(f"generation: {i}, IGD: {igd}")
            logger[i] = igd
        end = time.time()
        print("Generation {}, total time: {:4f}s".format(self.cfg.POPULATION.GENERATION ,end-since))
        self.plot_front(pop, fig_name='./final')
        self.plot_igd(logger, fig_name='./igd')
    
    def plot_front(self, pop, fig_name='.'):
        ranks = [individual.rank for individual in pop.individuals]
        front = [i for i in range(len(ranks)) if ranks[i]==0]
        obj_values = [pop.individuals[f].get_obj_values(self.problem.obj_funcs) for f in front]
        if len(self.problem.obj_funcs) == 2:
            p1 = plt.scatter([f[0] for f in obj_values], [f[1] for f in obj_values], marker='*', color="red")
            p2 = plt.scatter([f[0] for f in self.problem.truth_front], [f[1] for f in self.problem.truth_front], 
                marker='o', color="", edgecolors="green", s=20)
            plt.legend([p1, p2], ["Pareto front", "truth"])
            plt.ylabel("f2")
            plt.xlabel("f1")
            plt.title("{}".format(self.problem.name))
            plt.savefig("{}_{}.png".format(fig_name, self.problem.name))
            plt.close()
        elif len(self.problem.obj_funcs) == 3:
            from mpl_toolkits.mplot3d.axes3d import Axes3D
            fig = plt.figure()
            axes3d = fig.add_subplot(111, projection='3d')
            p1 = axes3d.scatter([f[0] for f in obj_values], [f[1] for f in obj_values], [f[2] for f in obj_values])
            p2 = axes3d.scatter([f[0] for f in self.problem.truth_front], [f[1] for f in self.problem.truth_front], [f[2] for f in self.problem.truth_front])
            axes3d.set_xlabel("f1")
            axes3d.set_ylabel("f2")
            axes3d.set_zlabel("f3")
            axes3d.legend([p1, p2], ["Pareto front", "truth"])
            axes3d.set_title("{}".format(self.problem.name))
            # plt.savefig("{}_{}.png".format(fig_name, self.problem.name))
            plt.show()
        else:
            raise ValueError("High object dimension can't be visualized!")
        
    
    def evaluate_igd(self, pop):
        ranks = [individual.rank for individual in pop.individuals]
        front = [i for i in range(len(ranks)) if ranks[i]==0]
        obj_values = np.array([pop.individuals[f].get_obj_values(self.problem.obj_funcs) for f in front])
        truth_front = np.array(self.problem.truth_front)
        len_tpf = truth_front.shape[0]
        sum_dist = 0
        for i in range(len_tpf):
            sum_dist += np.min(
                np.sqrt(
                    np.sum(
                        np.power(truth_front[i,:]-obj_values, 2), axis=1
                    )
                )
            )
        igd = sum_dist / len_tpf
        return igd
    
    def plot_igd(self, igds, fig_name='.'):
        plt.plot(igds)
        plt.ylabel('IGD')
        plt.xlabel("Generation")
        plt.title("{}-IGD".format(self.problem.name))
        plt.savefig("{}_{}.png".format(fig_name, self.problem.name))
        plt.close()
        

