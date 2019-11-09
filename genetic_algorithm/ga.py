import time
import numpy as np
import matplotlib.pyplot as plt

from .population import Population
 

class GA:
    def __init__(self, cfg):
        self.cfg = cfg
        self.obj_funcs = obj_funcs
        self.fitness_func = cfg["fitness_func"]
        self.top_k_reserved = cfg["top_k_reserved"]
        assert cfg.POPULATION.CODE_MODE == "binary", "Sorry, only implemented binary coded!"
        if isinstance(cfg.POPULATION.LENGTHS, int):
            self.cfg.POPULATION.LENGTHS = [cfg.POPULATION.LENGTHS] * cfg.OBJECT.DIMENSION
        else:
            assert isinstance(cfg.POPULATION.LENGTHS, (list, tuple))
        if isinstance(cfg.OBJECT.LOWER_BOUNDS, (int, float)):
            self.cfg.OBJECT.LOWER_BOUNDS = [cfg.OBJECT.LOWER_BOUNDS] * cfg.OBJECT.DIMENSION
        else:
            assert isinstance(cfg.OBJECT.LOWER_BOUNDS, (list, tuple))
        if isinstance(cfg.OBJECT.UP_BOUNDS, (int, float)):
            self.cfg.OBJECT.UP_BOUNDS = [cfg.OBJECT.UP_BOUNDS] * cfg.OBJECT.DIMENSION
        else:
            assert isinstance(cfg.OBJECT.UP_BOUNDS, (list, tuple))
    
    def run(self, save_dir="."):
        pop = Population(self.cfg, init_mode="size")
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
            offspring_pop = live_pop.crossover(mate_index, self.pc, self.cross_point)
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
        if self.optimal_direction == "maximize":
            best_index = np.argmax(best_y)
        elif self.optimal_direction == "minimize":
            best_index = np.argmin(best_y)
        else:
            raise ValueError("Invaild mode")
        if 'HD' in self.name:
            fig = plt.figure(figsize=(15, 8), dpi=80)
        else:
            fig = plt.figure(figsize=(10, 8), dpi=80)
        plt.boxplot(all_y)
        plt.plot(range(1, len(logger)+1), best_y)
        p1 = plt.scatter(best_index+1, best_y[best_index], marker='*', color='red', s=100)
        plt.legend([p1], ["{:.2f}".format(best_y[best_index])])
        plt.ylabel('obj_value')
        plt.xlabel("genetation")
        plt.title("{} {}".format(self.optimal_direction, self.name))
        plt.savefig("{}/{}.png".format(save_dir, self.name))
        plt.close()
