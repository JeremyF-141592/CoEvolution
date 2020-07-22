"""
FILE IS TO BE CLEANED AND REFACTORED

If you can read this warning, chances are the file will be unreadable and not useful.

"""
import matplotlib.pyplot as plt
import numpy as np
from Parameters import Configuration
from Utils.Stats import unpack_stats
from glob import glob
import re
import pickle

Configuration.make()

nnsga_folder = "./BenchmarkAnim/"

filenames = glob(f"{nnsga_folder}/*.pickle")
filenames = list(filter(lambda x: "Iteration" in x, filenames))
filenames.sort(key=lambda k: int(re.sub('\D', '', k)))

statos = unpack_stats(f"{nnsga_folder}/Stats.json")

# plt.show()
for t in range(0, len(filenames)):
    print(t)
    with open(filenames[t], "rb") as f:
        ea_load = pickle.load(f)

    pop_env = ea_load[1]
    env = pop_env[0]

    color = "cyan"
    legend = "Generalists"
    nnsga_ags = ea_load[2]

    # O t 1 0 -> Environment number 0 gives tuple (time, values), we choose the values with 1, at iteration t ajusted
    if t%20 < 10:
        color = "lime"
        legend = "Specialists"
        nnsga_ags = ea_load[0][0]
        good_iteration = statos["Objective_0-argmax"][0][0].index(t)
        best_ag = statos["Objective_0-argmax"][0][1][good_iteration]
    else:
        best_ag = 0

    if len(nnsga_ags) == 0:
        continue
    if len(nnsga_ags[0].get_weights()) != 2:
        raise AssertionError("Something went wrong, the agents are not 2 dimensional.")

    bounds = pop_env[0].bounds

    plt.clf()
    size = 100
    p_mean = -100
    k = np.linspace(bounds[0]*1.2, bounds[1]*1.2, num=size)
    a = np.zeros((size, size))
    full_points = np.vstack((env.generalist_points, env.specialist_points))
    full_fit = env.generalist_fit + env.specialist_fit
    for i in range(len(env.generalist_points)):
        plt.plot(env.generalist_points[i][0], env.generalist_points[i][1], "ob")
    for i in range(size):
        for j in range(size):
            fit = 0
            dist = 0
            for p in range(len(full_points)):
                d_inv = (np.linalg.norm(full_points[p] - np.array([k[i], k[j]])) + 1e-9)**-2
                fit += d_inv * full_fit[p]
                dist += d_inv
            fit /= dist
            a[j, i] = fit
    plt.imshow(a, cmap='hot', interpolation='nearest', extent=[bounds[0]*1.2, bounds[1]*1.2, bounds[1]*1.2, bounds[0]*1.2])

    if best_ag == 0:
        plt.plot(nnsga_ags[0].value[0], nnsga_ags[0].value[1], "*", color=color, label=legend)
    else:
        plt.plot(nnsga_ags[0].value[0], nnsga_ags[0].value[1], "o", color=color, label=legend)
    for i in range(1, len(nnsga_ags)):
        marker = "o"
        if i == best_ag:
            marker="*"
        plt.plot(nnsga_ags[i].value[0], nnsga_ags[i].value[1], marker, color=color)
    plt.plot(env.generalist_points[0][0], env.generalist_points[0][1], "ob", label="Generalist hill")

    if t % 20 == 0:
        plt.title(f"Iteration  {t} - MUTATION")
    else:
        plt.title(f"Iteration  {t}")
    # plt.legend()
    # plt.pause(0.01)
    plt.savefig(f"../anim/{t}")

