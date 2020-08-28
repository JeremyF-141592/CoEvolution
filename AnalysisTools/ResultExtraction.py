#!/usr/bin/env python
"""

Make plots out of dictionaries of test results, loaded from pickled files.

"""
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# path to pickled results
paths = ["../Results/FinalResults.pickle"]

bins = range(9)  # Bins for solvability histogram
group = True  # Group executions with the same radical followed by a number

# Rename dictionary keys
rename = {
    "NNSGA_f": "NNSGA Global PATA-EC (5)",
    "NSGA2_f": "NSGA2 Random Environments (5)",
    "NNSGA_3f": "NSGA2 Local + Global PATA-EC (5)",
    "NNSGA_4f": "NNSGA Global PATA-EC + Unique (1)",
    "POET_": "POET (5)"
}

# Fill this list to keep only the following keys
keep = [
    "POET_",
    "NNSGA_f",
    "NSGA2_f",
    "NNSGA_3f",
    "NNSGA_4f"
]

# Fill this list to remove the following keys
pop = [

]

# ----------------------------------------------------------------------------------------------------------------------
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

# KEEP
all_keys = list(batches.keys())
if len(keep) > 0:
    for p in all_keys:
        delete = True
        for s in keep:
            if re.match(s, p):
                delete = False
                break
        if delete and p in batches.keys():
            batches.pop(p)

# POP
all_keys = list(batches.keys())
for p in all_keys:
    delete = False
    for s in pop:
        if re.match(s, p):
            delete = True
            break
    if delete and p in batches.keys():
        batches.pop(p)

# RENAME
algorithm_batches = dict()
for k in batches.keys():
    if k in rename.keys():
        algorithm_batches[rename[k]] = batches[k].copy()
    else:
        algorithm_batches[k] = batches[k].copy()

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

gen = dict()
for key in algorithm_batches.keys():
    radical = re.sub("\d+$", "", str(key))
    if radical in gen.keys():
        gen[radical].append(solv[key])
    else:
        gen[radical] = [solv[key]]

for key in gen.keys():
    plt.title(f"Generalization score of the best agent - CollectBall ({nb_env} Test Environments)")
    for s in gen[key]:
        plt.plot(count, s, "ob", label=key)
        plt.text(count+0.05, s, round(s, 2))
    count += 1
    if key in rename.keys():
        labels.append(rename[key])
    else:
        labels.append(key)

sns.set_style('whitegrid')
plt.ylabel("Mean fitness over environments")
plt.xticks(np.arange(count), labels)  # Set text labels.
plt.xlim(0.8, count-0.8)
plt.show()
