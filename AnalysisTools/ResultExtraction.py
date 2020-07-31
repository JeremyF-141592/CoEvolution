import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = "../temp/Results.pickle"

with open(path, "rb") as f:
    res = pickle.load(f)

for key in res.keys():
    solvabilty = list()
    for ev in res[key]:
        scores = list()
        for i in range(len(ev)):
            scores.append(ev[i][0])
        print(len(scores))
        solvabilty.append(np.array(scores).max())
    points = [solvabilty.count(i) for i in range(6)]
    print(points)
    sns.set_style('whitegrid')
    plt.title("Solvability score distribution for each environment - CollectBall")
    plt.xlabel("Maximum Fitness")
    plt.ylabel("Environment count")
    plt.plot(points, "o", label=f"{key}")
    # sns.kdeplot(np.array(solvabilty), label=f"{key}")
    # plt.hist(solvabilty, bins=50, alpha=0.5)
    # plt.axvline(np.median(solvabilty), linestyle="--", color=color)
plt.legend()
plt.show()
