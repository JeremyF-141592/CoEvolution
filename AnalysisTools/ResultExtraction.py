import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

paths = ["../temp/ResultsJS.pickle"]
group = True
rename = {
    "NNSGA_JS": "NNSGA Mean + Euclidian Novelty + JS distance (1)",
    "NNSGA0_": "NNSGA Mean + Euclidian Novelty (1)"
}

res = dict()
for p in paths:
    with open(p, "rb") as f:
        res = {**pickle.load(f), **res}

if group:
    batches = dict()
    for key in res.keys():
        alg = re.sub("\d+$", "", str(key))
        print(alg)
        if alg in batches.keys():
            batches[alg] += res[key]
        else:
            batches[alg] = res[key]
else:
    batches = res

algorithm_batches = dict()
for k in batches.keys():
    if k in rename.keys():
        algorithm_batches[rename[k]] = batches[k]
    else:
        algorithm_batches[k] = batches[k]


for key in algorithm_batches.keys():
    solvabilty = list()
    for ev in algorithm_batches[key]:
        scores = list()
        for i in range(len(ev)):
            scores.append(ev[i][0])
        solvabilty.append(np.array(scores).max())
    print(key, ":", np.array(solvabilty).mean())
    points = [solvabilty.count(i) / len(solvabilty) for i in range(6)]
    sns.set_style('whitegrid')
    plt.title("Solvability score distribution - CollectBall (100 Test Environments)")
    plt.xlabel("Maximum Fitness")
    plt.ylabel("Test Environment proportion")
    plt.plot(points, label=f"{key}")
    # sns.kdeplot(np.array(solvabilty), label=f"{key}")
    # plt.hist(solvabilty, bins=50, alpha=0.5)
    # plt.axvline(np.median(solvabilty), linestyle="--", color=color)
plt.legend()
plt.show()

count = 1
for key in algorithm_batches.keys():
    solvabilty = list()
    scores = np.zeros((len(algorithm_batches[key]), len(algorithm_batches[key][0])))
    for i in range(len(algorithm_batches[key])):
        sc = list()
        for j in range(len(algorithm_batches[key][i])):
            sc.append(algorithm_batches[key][i][j][0])
        scores[i] = np.array(sc)
    scores = scores.T
    gen = list()
    for i in range(len(scores)):
        scores[i].sort()
        dot = np.arange(len(scores[i]), dtype=float) + 1
        dot = np.power(dot, -2)
        dot = dot / dot.sum()
        gen.append(np.multiply(scores[i], dot).sum())
    sns.set_style('whitegrid')
    plt.title("Generalization score distribution for all agents - CollectBall")
    plt.ylabel("Mean fitness over environments")
    plt.boxplot(gen, positions=[count], labels=[key])
    count += 1
    # sns.kdeplot(np.array(solvabilty), label=f"{key}")
    # plt.hist(solvabilty, bins=50, alpha=0.5)
    # plt.axvline(np.median(solvabilty), linestyle="--", color=color)
plt.legend()
plt.show()
