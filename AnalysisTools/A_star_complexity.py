"""
Plot some informations about A* complexity for the collectball environment, on an execution with multiple Iterations
saved. This file is rather ugly.
"""
from Parameters import Configuration
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
from glob import glob
import seaborn as sns
from Utils.Stats import min_max, mean_std
Configuration.make()

path = "temp/NNSGA_4f0/"  # path to the execution folder
name = "NNSGA_env"  # name of the saved pickled files
load = True  # wether to compute and save or load results


# ----------------------------------------------------------------------------------------------------------------------
pp = glob(path + "Iteration_*.pickle")

absc = list()
for p in pp:
    absc.append(int(re.search(".*_([0-9]+).*", p).group(1)))
absc = np.array(absc)

it, mini, maxi = min_max(path + "Stats.json", "Objective_general-0")
it2, mean, std = mean_std(path + "Stats.json", "Objective_general-0")

ev = Configuration.envFactory.new()

with open(path + f"Iteration_{absc.max()}.pickle", 'rb') as f:
    _, _, ags = pickle.load(f)

if not load:
    results = list()
    ag_scores = list()
    for i in range(len(pp)):
        results.append(list())
        ag_scores.append(list())
        with open(pp[i], 'rb') as f:
            _, pop_env, pop_gen = pickle.load(f)

        mean_diff = 0
        for ev in pop_env:
            ag_scores[-1].append(ev(ags[-1]))
            results[-1].append(ev.a_star_complexity())
        print("iteration", i)
    with open(name + "scores", "wb") as f:
        pickle.dump(ag_scores, f)
    with open(name + "results", "wb") as f:
        pickle.dump(results, f)

if load :
    with open(name + "scores", "rb") as f:
        ag_scores = pickle.load(f)

    with open(name + "results", "rb") as f:
        results = pickle.load(f)

for i in range(len(ag_scores)):
    for j in range(len(ag_scores[i])):
        ag_scores[i][j] = ag_scores[i][j][0]

sns.set_style('whitegrid')
plt.plot(maxi)
        # plt.plot(absc[i], results[i][j], "o")
plt.title("Maximum mean Fitness of active population during training")
plt.xlabel("Iterations")
plt.ylabel("Max. Mean Fitness of active population")
plt.show()

for i in range(len(pp)):
    for j in range(len(results[i])):
        plt.plot(absc[i], results[i][j], "o")
        # plt.plot(absc[i], results[i][j], "o")
plt.title("NNSGA CollectBall complexity (A* from goal to ball), PATA-EC + Uniques")
plt.ylabel("Complexity (A* from goal to ball)")
plt.xlabel("Iterations")
plt.show()

for i in range(len(pp)):
    for j in range(len(results[i])):
        plt.plot(results[i][j], maxi[absc[i]], "o")
        # plt.plot(absc[i], results[i][j], "o")
plt.title("Maximum Fitness of active population over complexity during training")
plt.xlabel("Complexity (A* from goal to ball)")
plt.ylabel("Max. Fitness of active population")
plt.show()

for i in range(len(pp)):
    for j in range(len(results[i])):
        plt.plot(results[i][j], mean[absc[i]], "o")
        # plt.plot(absc[i], results[i][j], "o")
plt.title("Mean Fitness of active population over complexity during training")
plt.xlabel("Complexity (A* from goal to ball)")
plt.ylabel("Mean Fitness of active population")
plt.show()
for i in range(len(pp)):
    for j in range(len(results[i])):
        plt.plot(results[i][j], ag_scores[i][j], "o")
        # plt.plot(absc[i], results[i][j], "o")
plt.title("Fitness over complexity of the last, best generalist")
plt.xlabel("Complexity (A* from goal to ball)")
plt.ylabel("Fitness of the last, best generalist")
plt.show()