from Environments.bipedal_walker_cppn import BipedalWalkerCPPN
from Environments.cppn import CppnEnvParams
import matplotlib.pyplot as plt
import pickle
import numpy as np
from Parameters import Configuration
from scipy.stats import linregress
from Utils.Stats import unpack_stats


def func(x):
    if x > 0.5:
        return 2*x - 1
    else:
        return -2*x + 1


Configuration.make()

with open("../Iteration_399.pickle", "rb") as f:
    it = pickle.load(f)

with open("../temp/Iteration_399.pickle", "rb") as f:
    it2 = pickle.load(f)

# it = it2
it = [it[0]]
original_scores = list()
diffs = list()
new_envs = list()

all_freqs = list()
all_offs = list()
count = 0
for ea_pair in it:
    E, theta = ea_pair
    score = E(theta)
    original_scores.append(score)
    print(score)

    all_freqs.append(E.benchmark_frequency)
    all_offs.append(E.benchmark_offset)
    count +=1

evs = [it[k][0].get_child() for k in range(len(it))]
for i in range(400):
    # sommation = np.random.uniform(0, 1, size=len(it))
    # # for k in range(len(sommation)):
    # #     sommation[k] = func(sommation[k])
    #
    # freq = 0
    # off = 0
    # for k in range(len(it)):
    #     freq += sommation[k] * it[k][0].benchmark_frequency
    #     off += sommation[k] * it[k][0].benchmark_offset
    # freq /= sommation.sum()
    # off /= sommation.sum()
    # ev = Configuration.baseEnv(Configuration.flatConfig)
    # ev.benchmark_frequency = freq
    # ev.benchmark_offset = off
    # new_envs.append(ev)
    for k in range(1):
        new_envs.append(evs[k])
        evs[k] = evs[k].get_child()

for i in range(len(it)):
    print(f"Pair n {i}")
    E, theta = it[i]
    diffs.append(list())

    print(f"{i} ----")
    print(E.benchmark_frequency.mean())
    print(E.benchmark_frequency.std())
    print(E.benchmark_offset.mean())
    print(E.benchmark_offset.std())
    for j in range(len(new_envs)):
        dist = E.benchmark_frequency + E.benchmark_offset - new_envs[j].benchmark_frequency - \
               new_envs[j].benchmark_offset
        # if abs(dist.mean()) > 0.1:
        diffs[i].append([np.linalg.norm(dist), new_envs[j](theta)])

diffs = np.array(diffs)
print(diffs.shape)
for i in range(len(it)):
    plt.plot(diffs[i, :, 0], diffs[i, :, 1], "+", label=f"{i}")
plt.title("Benchmark01 Variation in fitness over environment distance")
plt.ylabel("Raw fitness")
plt.xlabel("Optimum displacement")
plt.legend()
plt.show()

diffs = np.array(diffs)
print(diffs.shape)
for i in range(len(it)):
    slope, intercept, r_value, p_value, std_err = linregress(diffs[i, :, 0], diffs[i, :, 1])
    plt.plot(diffs[i, :, 0], slope*diffs[i, :, 0] + intercept, label=f"{i} - {round(r_value**2, 2):0.2f}")

ax = plt.axes()
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0][-3:]))
plt.title("Benchmark01 Variation in fitness over environment distance")
plt.ylabel("Raw fitness")
plt.xlabel("Optimum displacement")
ax.legend(handles, labels)
plt.show()

for i in range(len(it)):
    new_x = np.log(diffs[i, :, 0])
    slope, intercept, r_value, p_value, std_err = linregress(new_x, diffs[i, :, 1])
    plt.plot(new_x, slope*new_x + intercept, label=f"{i} - {round(r_value**2, 2):0.2f}")

ax = plt.axes()
handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0][-3:]))
plt.title("Benchmark01 Variation in fitness over environment distance (logx)")
plt.ylabel("Raw fitness")
plt.xlabel("Optimum displacement")
ax.legend(handles, labels)
plt.show()
