import math
import numpy as np

from .operator import fast_nondominated_sort

#TODO implement single object optimal function class.


class ZDT:
    def __init__(self, name, dimension):
        self.name = name
        self.dimension = dimension
        self.lower_bounds = [0] * self.dimension
        self.up_bounds = [1] * self.dimension
        g = lambda x, n=dimension: 1 + 9*(sum(x) - x[0]) / (n-1)
        f_1 = lambda x: x[0]
        if name == "ZDT1":
            f_2 = lambda x: g(x) * (1 - math.sqrt(x[0] / g(x)))
            self.obj_funcs = [f_1, f_2]
        elif name == "ZDT2":
            f_2 = lambda x: g(x) * (1 - math.pow(x[0] / g(x), 2))
            self.obj_funcs = [f_1, f_2]
        elif name == "ZDT3":
            f_2 = lambda x: g(x) * (1 - math.sqrt(x[0] / g(x)) - x[0] / g(x) * math.sin(10*math.pi*x[0]))
            self.obj_funcs = [f_1, f_2]
        # self.solutions = []
        self.optimal_direction = ["minimize"] * len(self.obj_funcs)
        x0 = np.arange(0, 1.001, 0.001)
        truth_x = [[x] + [0]*(self.dimension-1) for x in x0]
        truth_f = [[f_1(x), f_2(x)] for x in truth_x]
        _, truth_fronts = fast_nondominated_sort(truth_f, self.optimal_direction)
        self.truth_front = [[truth_f[f][0], truth_f[f][1]] for f in truth_fronts[0]]


class DTLZ:
    def __init__(self, name, dimension):
        self.name = name
        self.dimension = dimension
        self.lower_bounds = [0] * self.dimension
        self.up_bounds = [1] * self.dimension
        if name == "DTLZ1":
            # M = 3, n=7
            g = lambda x, n=dimension: 100*(len(x) - 2 + 
                np.sum(
                    (np.array(x)-0.5)**2 - np.cos(20*np.pi*(np.array(x)-0.5))
                ))
            f_1 = lambda x: 0.5*x[0]*x[1]*(1+g(x[2:]))
            f_2 = lambda x: 0.5*x[0]*(1-x[1])*(1+g(x[2:]))
            f_3 = lambda x: 0.5*(1-x[0])*(1+g(x[2:]))
            self.obj_funcs = [f_1, f_2, f_3]
        elif name == "DTLZ2":
            g = lambda x: np.sum((np.array(x) - 0.5)**2)
            f_1 = lambda x: np.cos(0.5*np.pi*x[0])*np.cos(0.5*np.pi*x[1])*(1+g(x[2:]))
            f_2 = lambda x: np.cos(0.5*np.pi*x[0])*np.sin(0.5*np.pi*x[1])*(1+g(x[2:]))
            f_3 = lambda x: np.sin(0.5*np.pi*x[1])*(1+g(x[2:]))
            self.obj_funcs = [f_1, f_2, f_3]
        self.optimal_direction = ["minimize"] * len(self.obj_funcs)
        x0 = np.arange(0, 1.1, 0.1)
        x1 = np.arange(0, 1.1, 0.1)
        truth_x = []
        for i in range(x0.size):
            for j in range(x1.size):
                truth_x.append([x0[i], x1[j]] + [0.5]*(self.dimension-2))
        truth_f = [[f_1(x), f_2(x), f_3(x)] for x in truth_x]
        _, truth_fronts = fast_nondominated_sort(truth_f, self.optimal_direction)
        self.truth_front = [[truth_f[f][0], truth_f[f][1], truth_f[f][2]] for f in truth_fronts[0]]
