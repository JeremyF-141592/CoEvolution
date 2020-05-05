import numpy as np
import json
from Parameters import Configuration


def agents_stats(agents):
    weights = np.zeros((len(agents), len(agents[0].get_weights())))
    for i in range(len(agents)):
        weights[i] = agents[i].get_weights()
    #  Get mean distance between agents
    dist_mean = 0
    for i in range(len(weights)):
        for j in range(i, len(weights)):
            dist_mean += np.linalg.norm(weights[i] - weights[j])
    dist_mean /= (len(weights) * len(weights+1))/2
    w_mean = weights.mean(axis=0).tolist()
    return float(dist_mean), w_mean


def raw_fitness(agents, envs):
    res = list()
    for i in range(len(agents)):
        mini_res = 0
        for k in range(5):
            mini_res += envs[i](agents[i])
        res.append(mini_res / 5.0)
    return res


def benchmark_evolution(envs):
    res = list()
    for i in range(len(envs)):
        res.append((envs[i].benchmark_offset.sum()-len(envs[i].benchmark_offset)) * envs[i].argmax + envs[i].benchmark_offset.sum())
    return res


def bundle_stats(agents, envs):
    dic = dict()
    ag_stats = agents_stats(agents)
    dic["Dist Mean"] = ag_stats[0]
    dic["Weight Mean"] = ag_stats[1]
    dic["Fitness"] = raw_fitness(agents, envs)
    if Configuration.benchmark is not None:
        dic["Benchmark"] = benchmark_evolution(envs)
    return dic


def append_stats(path, bundle):
    with open(path, "a") as f:
        f.write(json.dumps(bundle))
        f.write("\n")


def unpack_stats(path):
    res = list()
    with open(path, "r") as f:
        for line in f.readlines():
            dic = json.loads(line)
            res.append(dic)
    return res
