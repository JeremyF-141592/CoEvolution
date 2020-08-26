#!/usr/bin/env python
"""

Make plots out of dictionaries of test results, loaded from pickled files.

"""
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

paths = ["../temp/ResultsFinal.pickle"]
bins = range(9)
group = True
rename = {
    "NNSGA_JSG": "NNSGA Global + JS (5)",
    "NNSGA_G": "NNSGA Global (5)",
    "NNSGA_": "NNSGA Global + Local (5)",
    # "NNSGA20K_": "NNSGA 12K Local + Global (1)",
    "POET_": "POET (5)"
}
pop = [
    "NNSGA_JS",
    "NNSGA20K_",
    "NNSGA20K_G"
]

res = dict()
for p in paths:
    with open(p, "rb") as f:
        res = {**pickle.load(f), **res}
nb_env = len(res[list(res.keys())[0]])
if group:
    batches = dict()
    for key in res.keys():
        alg = re.sub("\d+$", "", str(key))
        print(alg)
        if alg in batches.keys():
            batches[alg] += res[key].copy()
        else:
            batches[alg] = res[key].copy()
else:
    batches = res.copy()

algorithm_batches = dict()
for k in batches.keys():
    if k in rename.keys():
        algorithm_batches[rename[k]] = batches[k].copy()
    else:
        algorithm_batches[k] = batches[k].copy()

for p in pop:
    algorithm_batches.pop(p)

print("--- Mean solvability ---")
sns.set_style('whitegrid')
for key in algorithm_batches.keys():
    solvabilty = list()
    for ev in algorithm_batches[key]:
        scores = list()
        for i in range(len(ev)):
            scores.append(ev[i][0])
        solvabilty.append(np.array(scores).max())
    print("\t", key, ":", np.array(solvabilty).mean())
    points = np.histogram(solvabilty, bins=bins, density=True)[0]
    plt.title(f"Solvability score distribution - CollectBall ({nb_env} Test Environments)")
    plt.plot(points, label=f"{key}")

plt.xlabel("Maximum Fitness")
plt.ylabel("Test Environment proportion")
plt.legend()
plt.show()

labels = [" "]
count = 1
solv0 = dict()
for key in res.keys():
    solvabilty = list()
    scores = np.zeros((len(res[key]), len(res[key][0])))
    for i in range(len(res[key])):
        sc = list()
        for j in range(len(res[key][i])):
            sc.append(res[key][i][j][0])
        scores[i] = np.array(sc)
    scores = scores.T
    gen = list()
    for i in range(len(scores)):
        gen.append(scores[i].mean())
    if group:
        alg = re.sub("\d+$", "", str(key))
        if alg in solv0.keys():
            solv0[alg] = max(solv0[alg], np.array(gen).max())
        else:
            solv0[alg] = np.array(gen).max()
    else:
        solv0[key] = np.array(gen).max()
solv = dict()
for k in solv0.keys():
    if k in rename.keys():
        solv[rename[k]] = solv0[k]
    else:
        solv[k] = solv0[k]

for key in algorithm_batches.keys():
    plt.title(f"Generalization score of the best agent - CollectBall ({nb_env} Test Environments)")
    plt.plot(count, solv[key], "ob", label=key)
    plt.text(count+0.05, solv[key], round(solv[key], 2))
    count += 1
    labels.append(key)

sns.set_style('whitegrid')
plt.ylabel("Mean fitness over environments")
plt.xticks(np.arange(count), labels)  # Set text labels.
# plt.legend()
plt.show()
