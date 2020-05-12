from Environments.bipedal_walker_cppn import BipedalWalkerCPPN
from Environments.cppn import CppnEnvParams
import matplotlib.pyplot as plt
import pickle
import numpy as np
from Parameters import Configuration
from Utils.Stats import unpack_stats

stats = list()
for i in range(1, 4):
    stats.append(unpack_stats(f"temp/Stats{i}.json"))

for k in range(len(stats)):
    for tup in stats[k]["Dist_Mean"]:
        plt.plot(tup[0], tup[1])
    plt.title("Mean agent distance")
    plt.ylabel("Euclidean Distance")
    plt.xlabel("Iterations")
    for i in range(40, 400, 20):
        plt.axvline(i, ls=":", color="black")
    plt.show()

for k in range(len(stats)):
    for tup in stats[k]["Benchmark"]:
        plt.plot(tup[0], tup[1])
    plt.title("Environment diversity")
    for i in range(40, 400, 20):
        plt.axvline(i, ls=":", color="black")
    plt.show()

for k in range(len(stats)):
    for tup in stats[k]["Fitness"]:
        plt.plot(tup[0], tup[1])
    plt.title("Agent fitness")
    for i in range(40, 400, 20):
        plt.axvline(i, ls=":", color="black")
    plt.axhline(40, ls=":", color="red")
    plt.ylabel("Raw Fitness")
    plt.xlabel("Iterations")
    plt.show()