import functions as f
import pso
import bo 
import numpy as np
import math
import matplotlib.pyplot as plt
import tqdm
from multiprocessing import Pool

def main():

    function_list = ["Ackley", "Rastgrin", "Sphere", "Rosenbrock", "Booth", "Easom"]

    with Pool(6) as pool:
        for function in tqdm.tqdm(function_list, desc="Benchmarking each function"):
            z, x, y, a, fx = f.create_benchmark(function)
            # paralellize run and get 50 runs of each algorithm and plot the average result  for each iteration from 5 10 15 30 60 100
            average_pso = []
            average_bo = []
            iters = [10, 15, 30, 60, 100]
            for iter in tqdm.tqdm(iters, desc="Benchmarking each iteration"):
                average_pso.append(np.mean(pool.starmap(pso.pso, [(a, fx, z, x, y, iter)]*50)))
                average_bo.append(np.mean(pool.starmap(bo.bo, [(fx, iter)]*50)))
            # store the average result for each iteration from 5 10 15 30 60 100
            np.save(f"data/average_pso_{function}", average_pso)
            np.save(f"data/average_bo{function}", average_bo)

            plt.plot(iters, average_pso, label="PSO")
            plt.plot(iters, average_bo, label="BO")
            plt.xlabel("Iterations")
            plt.ylabel("Average {} value".format(function))
            plt.title("Average {} value for each iteration".format(function))
            plt.legend()
            # save plot as png
            plt.savefig("plots/{}.png".format(function))
            plt.close()

if __name__ == "__main__":
    main()