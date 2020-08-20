"""
Perform an animation of the simple 2D benchmark by representing agents as red dots on the 2D fitness map.
"""

import matplotlib.pyplot as plt
from Utils.Stats import unpack_stats
from Objects.Environments.Benchmark2D import *

Configuration.make()


benchmark_function = diag_gaussian
bounds = [-50, 50, -5, 25]
stats = unpack_stats("../benchmark_og/Stats.json")


env_ogs = None
# plt.show()
for iteration in range(0, 300):
    plt.clf()
    plt.title("Iteration {}".format(iteration))

    env_ogs = list()
    for k in range(len(stats['2D_benchmark'])):
        if iteration in stats['2D_benchmark'][k][0]:
            ajusted = iteration - stats['2D_benchmark'][k][0][0]
            env_ogs.append(stats['2D_benchmark'][k][1][ajusted][1])

    env_ogs = np.unique(env_ogs)

    ags = list()
    for k in range(len(stats['2D_benchmark'])):
        if iteration in stats['2D_benchmark'][k][0]:
            ajusted = iteration - stats['2D_benchmark'][k][0][0]
            ags.append(stats['2D_benchmark'][k][1][ajusted])

    cal = np.array(env_ogs)
    # full_ev += evs

    size = 200
    k = np.linspace(bounds[0], bounds[1], num=size)
    a = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            a[j, i] = benchmark_function(k[i], k[j])

    plt.imshow(a, cmap='Greys', interpolation='nearest', extent=[bounds[0], bounds[1], bounds[2], bounds[3]])
    plt.xlim(bounds[0], bounds[1])
    plt.ylim(bounds[2], bounds[3])
    for env_og in env_ogs:
        plt.axhline(env_og, linestyle="dashed")

    for a in ags:
        plt.plot(a[0], a[1], "or")
    plt.xlabel("Agent")
    plt.ylabel("Environment")
    plt.pause(0.01)
    # plt.savefig(f"../anim2/{iteration}.png")
