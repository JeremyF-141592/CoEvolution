import matplotlib.pyplot as plt
import numpy as np
from Parameters import Configuration
from Environments.KNN_Benchmark import print_points
from glob import glob
import re
import pickle

Configuration.make()

envs_file = "../Test_Environments.pickle"
poet_file = "../POET_ag.pickle"
nnsga_file = "../NNSGA_ag.pickle"

with open(envs_file, "rb") as f:
    envs = pickle.load(f)
with open(poet_file, "rb") as f:
    poet_ags = pickle.load(f)
with open(nnsga_file, "rb") as f:
    nnsga_ags = pickle.load(f)

print(len(poet_ags))
print(len(nnsga_ags))
plt.show()
for env in envs:
    if len(poet_ags[0].get_weights()) != 2:
        raise AssertionError("Something went wrong, the agents are not 2 dimensional.")

    full_points = np.vstack((env.generalist_points, env.specialist_points))
    full_fit = env.generalist_fit + env.specialist_fit
    bounds = env.bounds

    plt.clf()
    plt.plot(poet_ags[0].value[0], poet_ags[0].value[1], "og", label="POET agents")
    for ag in poet_ags[1:]:
        plt.plot(ag.value[0], ag.value[1], "og")
    plt.plot(nnsga_ags[0].value[0], nnsga_ags[0].value[1], "oy", label="NNSGA agents")
    for ag in nnsga_ags[1:]:
        plt.plot(ag.value[0], ag.value[1], "oy")
    plt.plot(env.generalist_points[0][0], env.generalist_points[0][1], "ob", label="Generalist hill")
    print_points(full_points, full_fit, [-2, 2], cut=len(env.generalist_points))
    plt.title("Example of a 2D test environment - budget : 20k evaluations")
    plt.legend()
    plt.pause(2)

