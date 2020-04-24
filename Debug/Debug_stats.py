from Parameters import Configuration
from Utils import Stats
import numpy as np
import matplotlib.pyplot as plt


Configuration.make()

# Resume execution -----------------------------------------------------------------------------------------------------

folder = "../POET_execution"

if folder != "":

    stats = Stats.unpack_stats(f"{folder}/Stats.json")

    ag_matrix = np.zeros((len(stats), len(stats[0]["raw"]), len(stats[0]["raw"][0])))
    print(ag_matrix.shape)
    for i in range(len(stats)):
        ag_matrix[i] = stats[i]["raw"]
    mean_dist = np.zeros(len(ag_matrix))
    for i in range(len(ag_matrix)):
        mdist = 0
        for j in range(len(ag_matrix[i])):
            for k in range(j, len(ag_matrix[i])):
                mdist += np.linalg.norm(ag_matrix[i, j] - ag_matrix[i, k])
        mdist /= len(ag_matrix[i])*(len(ag_matrix[i])+1)*0.5
        mean_dist[i] = mdist
    plt.axhline(y=0, color="black")
    plt.axvline(x=15, linestyle="dashed", color="orange")
    plt.axvline(x=30, linestyle="dashed", color="orange")
    plt.axvline(x=45, linestyle="dashed", color="orange")
    for i in range(50, len(ag_matrix), 10):
        plt.axvline(x=i, linestyle="dashed", color="orange")
    plt.plot(mean_dist)
    plt.show()
