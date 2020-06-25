from Environments.bipedal_walker_cppn import BipedalWalkerCPPN
from Environments.cppn import CppnEnvParams
import matplotlib.pyplot as plt
import pickle
import numpy as np
from Parameters import Configuration
from Utils.Stats import unpack_stats
from Utils.Benchmark import *


Configuration.make()

bench = diag_gaussian

t_local = 15
t_global = 15

archive = list()
full_ev2 = list()
env_ogs = None
stats = unpack_stats("./NNSGA_stats.json")


plt.show()
for iteration in range(0, 300):
    local = iteration % (t_local + t_global) < t_local

    plt.clf()
    plt.title("Iteration {} {:>5}".format("locale" if local else "globale", iteration))
    print(iteration)

    env_ogs = list()
    for k in range(len(stats['xy_benchmark'])):
        if iteration in stats['xy_benchmark'][k][0]:
            ajusted = iteration - stats['xy_benchmark'][k][0][0]
            env_ogs.append(stats['xy_benchmark'][k][1][ajusted][1])

    ags = list()
    for k in range(len(stats['xy_benchmark'])):
        if iteration in stats['xy_benchmark'][k][0]:
            ajusted = iteration - stats['xy_benchmark'][k][0][0]
            ags.append(stats['xy_benchmark'][k][1][ajusted][0])

    cal = np.array(env_ogs)
    # full_ev += evs
    bounds = [-50, 50, -5, 25]

    size = 200
    k = np.linspace(bounds[0], bounds[1], num=size)
    a = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            a[j, i] = bench(k[i], k[j])

    plt.imshow(a, cmap='Greys', interpolation='nearest', extent=[bounds[0], bounds[1], bounds[2], bounds[3]])
    for env_og in env_ogs:
        plt.axhline(env_og, linestyle="dashed")
    for a in range(len(ags)):
        plt.plot(ags[a], env_ogs[a], "or")
    plt.xlabel("Agent")
    plt.ylabel("Environment")
    plt.pause(0.01)
    plt.savefig(f"../anim1/{iteration}.png")
