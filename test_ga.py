import math
import numpy as np
from genetic_algorithm.ga import GA


def objective_1():
    cfg = {}
    cfg["name"] = "Sphere-2D"
    cfg["dimension"] = 2
    cfg["obj_func"] = lambda x: np.sum(np.power(x, 2))
    cfg["fitness_func"] = lambda x: np.sum(np.power(x, 2))
    cfg["lower_bounds"] = -5.12
    cfg["up_bounds"] = 5.12
    cfg["optimal_direction"] = "maximize"
    return cfg

def objective_2():
    cfg = {}
    cfg["name"] = "De-Jong"
    cfg["dimension"] = 2
    cfg["obj_func"] = lambda x: 100*(x[1] - x[0]**2)**2 + (x[0]-1)**2
    cfg["fitness_func"] = lambda x: 100*(x[1] - x[0]**2)**2 + (x[0]-1)**2
    cfg["lower_bounds"] = -2.048
    cfg["up_bounds"] = 2.048
    cfg["optimal_direction"] = "maximize"
    return cfg

def objective_3():
    cfg = {}
    cfg["name"] = "Himmelbaut"
    cfg["dimension"] = 2
    cfg["obj_func"] = lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    cfg["fitness_func"] = lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    cfg["lower_bounds"] = -6
    cfg["up_bounds"] = 6
    cfg["optimal_direction"] = "maximize"
    return cfg

def objective_4():
    cfg = {}
    cfg["name"] = "Sphere-HD"
    cfg["dimension"] = 20
    cfg["obj_func"] = lambda x: np.sum(np.power(x, 2))
    cfg["fitness_func"] = lambda x: np.sum(np.power(x, 2))
    cfg["lower_bounds"] = -100
    cfg["up_bounds"] = 100
    cfg["optimal_direction"] = "maximize"
    return cfg

def objective_5():
    cfg = {}
    cfg["name"] = "Step-HD"
    cfg["dimension"] = 20
    cfg["obj_func"] = lambda x: np.sum(np.power(np.round(np.array(x)+0.5), 2))
    cfg["fitness_func"] = lambda x: np.sum(np.power(np.round(np.array(x)+0.5), 2))
    cfg["lower_bounds"] = -100
    cfg["up_bounds"] = 100
    cfg["optimal_direction"] = "maximize"
    return cfg

def objective_6():
    cfg = {}
    cfg["name"] = "Schwefel-HD"
    cfg["dimension"] = 20
    cfg["obj_func"] = lambda x: np.sum(np.array(x) + np.absolute(x))
    cfg["fitness_func"] = lambda x: np.sum(np.array(x) + np.absolute(x))
    cfg["lower_bounds"] = -10
    cfg["up_bounds"] = 10
    cfg["optimal_direction"] = "maximize"
    return cfg

def low_dimension_single_objective_optimization():
    cfg = {}
    cfg["generation"] = 20
    cfg["top_k_reserved"] = 2
    cfg["population_size"] = 10
    cfg["lengths"] = 8
    cfg["num_point"] = 1
    cfg["pc"] = 0.6
    cfg["pm"] = 0.1
    cfg["code_mode"] = "binary"
    return cfg


def high_dimension_single_objective_optimization():
    cfg = {}
    cfg["generation"] = 20
    cfg["top_k_reserved"] = 2
    cfg["population_size"] = 10
    cfg["lengths"] = 8
    cfg["num_point"] = 1
    cfg["pc"] = 0.6
    cfg["pm"] = 0.1
    cfg["code_mode"] = "binary"
    return cfg


def main():
    # low/high dimension single objective optimization
    cfg = low_dimension_single_objective_optimization()
    # cfg = high_dimension_single_objective_optimization()
    for obj in [objective_1, objective_2, objective_3]:
        cfg.update(obj())
        ga = GA(cfg)
        ga.run(save_dir="result")

    
    cfg = low_dimension_single_objective_optimization()
    # cfg = high_dimension_single_objective_optimization()
    for obj in [objective_4, objective_5, objective_6]:
        cfg.update(obj())
        ga = GA(cfg)
        ga.run(save_dir="result")


if __name__ == "__main__":
    main()