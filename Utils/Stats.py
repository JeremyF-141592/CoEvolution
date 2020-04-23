import numpy as np
import json


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
    cov = np.cov(weights).tolist()
    return dist_mean, w_mean, cov


def bundle_stats(agents, envs):
    dic = dict()
    dic["raw"] = list()
    # ag_stats = agents_stats(agents)
    # dic["Dist Mean"] = ag_stats[0]
    # dic["Weight Mean"] = ag_stats[1]
    # dic["Covariance"] = ag_stats[2]
    for i in range(len(agents)):
        dic["raw"].append(agents[i].get_weights().tolist())
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
