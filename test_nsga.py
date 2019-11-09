from genetic_algorithm.nsga import NSGA
from genetic_algorithm.problem import ZDT, DTLZ
from genetic_algorithm.config import _C as cfg


def main():
    # problem = ZDT(name="ZDT3", dimension=30)
    problem = DTLZ(name="DTLZ1", dimension=7)
    nsga2 = NSGA(cfg, problem)
    nsga2.run()


if __name__ == "__main__":
    main()