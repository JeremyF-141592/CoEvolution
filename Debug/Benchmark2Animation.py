import matplotlib.pyplot as plt
import numpy as np
from Parameters import Configuration
from Environments.KNN_Benchmark import print_points
from glob import glob
import re
import pickle

Configuration.make()

envs_file = "Test_Environments.pickle"
poet_file = "POET_ag.pickle"
nnsga_file = "NNSGA_ag.pickle"

with open(envs_file, "rb") as f:
    envs = pickle.load(f)
with open(poet_file, "rb") as f:
    poet_ags = pickle.load(f)
with open(nnsga_file, "rb") as f:
    nnsga_ags = pickle.load(f)

plt.show()
for env in envs:
    if len(poet_ags[0].get_weights()) != 2:
        raise AssertionError("Something went wrong, the agents are not 2 dimensional.")

    full_points = env.generalist_points + env.specialist_points
    full_fit = env.generalist_fit + env.specialist_fit
    bounds = env.bounds

    plt.clf()
    print_points(full_points, full_fit, bounds)
    for ag in poet_ags:
        plt.plot(ag.value[0], ag.value[1], "og")
    for ag in nnsga_ags:
        plt.plot(ag.value[0], ag.value[1], "oy")
    plt.pause(0.25)

