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
        res.append(envs[i].y_value)
    return res


def bundle_stats(agents, envs):
    dic = dict()
    ag_stats = agents_stats(agents)
    dic["Dist_Mean"] = ag_stats[0]
    # dic["Weight Mean"] = ag_stats[1]
    if len(agents) == len(envs):
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

    keys = res[0].keys()
    end_res = dict()
    for key in keys:
        value = []
        absciss = []
        for i in range(len(res)):
            if type(res[i][key]) != list:
                res[i][key] = [res[i][key]]
            if len(res[i][key]) > len(value):
                for add in range(len(res[i][key]) - len(value)):
                    value.append([])
                    absciss.append([])
            for j in range(len(res[i][key])):
                value[j].append(res[i][key][j])
                absciss[j].append(i)
        end_res[key] = list()
        for k in range(len(value)):
            end_res[key].append((absciss[k], value[k]))
    return end_res
