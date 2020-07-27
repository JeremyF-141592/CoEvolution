import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
sys.path.insert(1, "../")

from Utils.Stats import unpack_stats


def plot_raw(key, dic, destination):
    """Plot raw stats as they were saved, ex. an objective for every single agent."""
    for tup in dic[key]:
        extended_ord = np.empty((tup[0][-1]))
        extended_ord[:] = np.nan
        for i in range(len(tup[0])):
            extended_ord[tup[0][i]] = tup[1][i]
        plt.plot(extended_ord)
    plt.savefig(f"{destination}/Raw_{key}")


def plot(value, dic, destination):
    extensions = list()
    max_range = 0
    for tup in value:
        if tup[0][-1] > max_range:
            max_range = tup[0][-1] + 1
        extended_ord = np.empty((tup[0][-1]+1))
        extended_ord[:] = np.nan
        for i in range(len(tup[0])):
            extended_ord[tup[0][i]] = tup[1][i]
        extensions.append(extended_ord)
        plt.plot(extended_ord)  # Raw stats
    plt.title(f"{key} raw data")
    plt.savefig(f"{destination}/Raw_{key}")
    plt.clf()
    # Min and Max
    max_ext = np.empty(max_range)
    max_ext[:] = np.nan
    min_ext = np.empty(max_range)
    min_ext[:] = np.nan

    cat = [list() for i in range(max_range)]
    for ext in extensions:
        for i in range(len(ext)):
            if ext[i] != np.nan:
                cat[i].append(ext[i])

    med_ext = np.empty(max_range)
    med_ext[:] = np.nan
    quart1 = np.empty(max_range)
    quart1[:] = np.nan
    quart3 = np.empty(max_range)
    quart3[:] = np.nan
    for i in range(len(cat)):
        if len(cat[i]) != 0:
            val = np.array(cat[i])
            med_ext[i] = np.quantile(val, 0.5)
            quart1[i] = np.quantile(val, 0.25)
            quart3[i] = np.quantile(val, 0.75)
            max_ext[i] = np.quantile(val, 1.0)
            min_ext[i] = np.quantile(val, 0.0)

    plt.fill_between(np.arange(max_range), quart3, quart1, color=(0, 0.5, 1, 0.5))
    plt.plot(np.arange(max_range), med_ext, "r")
    plt.title(f"{key} median and quartiles")
    plt.savefig(f"{destination}/Med_{key}")
    plt.clf()

    plt.plot(max_ext)
    plt.plot(min_ext)
    plt.title(f"{key} minimum and maximum")
    plt.savefig(f"{destination}/MinMax_{key}")
    plt.clf()


# path = input("Path to execution folder :")
# while not os.path.exists(path) and not os.path.isdir(path):
#     path = input("Path to execution folder :")
path = "../temp/CollectBallNNSGA_L"
dic = unpack_stats(f"{path}/Stats.json")
if not os.path.exists(f"{path}/Plots"):
    os.mkdir(f"{path}/Plots")
for key, value in dic.items():
    plot(value, key, f"{path}/Plots")
    plt.close()

print("Done.")
