import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(1, "../")

from Utils.Stats import unpack_stats, mean_std
from Parameters import Configuration

Configuration.make()

file_selected = False
stat = dict()
path = ""

folder_mode = False

print("Type exit to exit, + after your key choice for a mean+std plot. \n")
while True:
    if not file_selected:
        path = input("Path to stat file : ")
        if path == "exit":
            break
        if not os.path.exists(path):
            continue
        stat = unpack_stats(path)
        file_selected = True

    k = np.linspace(-20, 20, num=100)
    a = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            a[j, i] = Configuration.benchmark(k[i], k[j])

    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.show()
    for i in range(0, len(stat["x_benchmark"][0][0]), 5):
        plt.clf()
        for j in range(len(stat["x_benchmark"])):
            plt.plot(stat["x_benchmark"][j][1][i]+20, stat["y_benchmark"][j][1][i]+20, "o")
        plt.imshow(a, cmap='hot', interpolation='nearest', origin='lower')
        plt.title(f"Iteration {i}")
        plt.pause(0.01)

