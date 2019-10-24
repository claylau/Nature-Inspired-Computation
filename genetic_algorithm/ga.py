import time
import matplotlib.pyplot as plt

from .population import Individual, Population
 

class GA:
    def __init__(self, cfg):
        self.cfg = cfg
        self.generation = cfg["generation"]
        self.dimension = cfg["dimension"]
        self.obj_func = cfg["obj_func"]
        self.fitness_func = cfg["fitness_func"]
        self.lower_bounds = cfg["lower_bounds"]
        self.up_bounds = cfg["up_bounds"]
        self.pop_size = cfg["population_size"]
        self.lengths = cfg["lengths"]
        self.num_point = cfg["num_point"]
        self.pc = cfg["pc"]
        self.pm = cfg["pm"]
        self.code_mode = cfg["code_mode"]
        self.optimal_direction = cfg["optimal_direction"]
        self.name = cfg["name"]
        self.top_k_reserved = cfg["top_k_reserved"]
        assert self.code_mode == "binary", "Sorry, only implemented binary coded!"
        if isinstance(self.lengths, int):
            self.lengths = [self.lengths] * self.dimension
        else:
            assert isinstance(self.lengths, (list, tuple))
        if isinstance(self.lower_bounds, (int, float)):
            self.lower_bounds = [self.lower_bounds] * self.dimension
        else:
            assert isinstance(self.lower_bounds, (list, tuple))
        if isinstance(self.up_bounds, (int, float)):
            self.up_bounds = [self.up_bounds] * self.dimension
        else:
            assert isinstance(self.up_bounds, (list, tuple))
    
    def run(self, save_dir="."):
        pop = Population(init_mode="size", size=self.pop_size, dimension=self.dimension,
            lower_bounds=self.lower_bounds, up_bounds=self.up_bounds, code_mode=self.code_mode,
            lengths=self.lengths)
        logger = list(range(self.generation))
        logger_file = open(f"{save_dir}/{self.name}.log", "w")
        print(self.cfg, file=logger_file)
        result = pop.evaluation(self.obj_func, self.optimal_direction)
        logger[0] = result
        print('generation 0: best_x:{}, best_y:{}'.format(result["best_x"], result["best_y"]), file=logger_file)
        since = time.time()
        for i in range(1, self.generation):
            live_pop, _ = pop.environment_selection(self.fitness_func, self.pop_size)
            mate_index = live_pop.mate_selection(self.fitness_func)
            offspring_pop = live_pop.crossover(mate_index, self.pc, self.num_point)
            offspring_pop.mutation(self.pm)
            offspring_pop.update_solution()
            elite_pop = pop.elite_selection(self.fitness_func, self.top_k_reserved)
            pop = offspring_pop + elite_pop
            result = pop.evaluation(self.obj_func, self.optimal_direction)
            logger[i] = result
            print('generation {}: best_x:{}, best_y:{}'.format(i, result["best_x"], result["best_y"]), file=logger_file)
        end = time.time()
        print("Total time: {:.4f}s".format(end-since), file=logger_file)
        self.plot_log(logger, save_dir)
    
    def plot_log(self, logger, save_dir):
        all_y = [r["y"] for r in logger]
        best_y = [r["best_y"] for r in logger]
        plt.boxplot(all_y)
        plt.plot(range(1, len(logger)+1), best_y)
        plt.ylabel('obj_value')
        plt.xlabel("genetation")
        plt.title("{} {}".format(self.optimal_direction, self.name))
        plt.savefig("{}/{}.png".format(save_dir, self.name))
        plt.close()
