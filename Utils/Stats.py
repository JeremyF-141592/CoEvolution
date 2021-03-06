"""
Stats are made by saving a dict "name_of_stat" : list(float) at each iteration
doing so requires the choice of a set of information to be saved, each stat is therefore defined by a function of
agents and environments at a given iteration.

This way of saving makes it so that there might be different size of stats on consecutive iterations, combining
the information back together requires a bit of manipulation.
"""

import numpy as np
import json


# Measure functions ----------------------------------------------------------------------------------------------------
def mean_ag_dist(agents, envs):
    weights = np.zeros((len(agents), len(agents[0].get_weights())))
    for i in range(len(agents)):
        weights[i] = agents[i].get_weights()
    #  Get mean distance between agents
    dist_mean = 0
    for i in range(len(weights)):
        for j in range(i, len(weights)):
            dist_mean += np.linalg.norm(weights[i] - weights[j])
    dist_mean /= (len(weights) * len(weights+1))/2
    return float(dist_mean)


def paired_fitness(agents, envs):
    res = list()
    for i in range(len(agents)):
        mini_res = 0
        for k in range(5):
            a = envs[i](agents[i])
            if type(a) == tuple or type(a) == list:
                mini_res += envs[i](agents[i])[0]
            else:
                mini_res += envs[i](agents[i])
        res.append(mini_res / 5.0)
    return res


def benchmark_evolution(ags, envs):
    res = list()
    for i in range(len(envs)):
        for j in range(len(ags)):
            res.append((ags[j].value, envs[i].y_value))
    return res


# Stats bundle manipulation --------------------------------------------------------------------------------------------
def unpack_stats(path):
    """Reads a pickled stat bundle and returns an unique dictionary of stats, containing for each
    stats tuples of the form (iteration_number, corresponding_stat_value). Therefore, if one information was
    kept only at iteration 50, there will be a tuple (50, the information)."""
    res = list()
    with open(path, "r") as f:
        for line in f.readlines():
            dic = json.loads(line)
            res.append(dic)

    keys = list(res[0].keys())
    for i in range(len(res)):
        for key in res[i].keys():
            if key not in keys:
                keys.append(key)
    end_res = dict()
    for key in keys:
        value = []
        absciss = []
        for i in range(len(res)):
            if key not in res[i]:
                continue
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


def mean_std(path, key):
    """Reads a pickled stat bundle, and return (iterations, mean, std) of a stat."""
    res = list()
    with open(path, "r") as f:
        for line in f.readlines():
            dic = json.loads(line)
            res.append(dic)

    value1 = []
    value2 = []
    absciss = []
    for i in range(len(res)):
        if key not in res[i]:
            continue
        if type(res[i][key]) != list:
            res[i][key] = [res[i][key]]

        me = np.array(res[i][key])
        value1.append(me.mean())
        value2.append(me.std())
        absciss.append(i)
    return np.array(absciss), np.array(value1), np.array(value2)


def med_qtile(path, key):
    """Reads a pickled stat bundle, and return (iterations, median, 1 quartile, 3 quartile) of a stat."""
    res = list()
    with open(path, "r") as f:
        for line in f.readlines():
            dic = json.loads(line)
            res.append(dic)

    value1 = []
    value2 = []
    value3 = []
    absciss = []
    for i in range(len(res)):
        if key not in res[i]:
            continue
        if type(res[i][key]) != list:
            res[i][key] = [res[i][key]]

        me = np.array(res[i][key])
        value1.append(np.median(me))
        value2.append(np.quantile(0.25))
        value3.append(np.quantile(0.75))
        absciss.append(i)
    return np.array(absciss), np.array(value1), np.array(value2), np.array(value3)


def min_max(path, key):
    """Reads a pickled stat bundle, and return (iterations, min, max) of a stat."""
    res = list()
    with open(path, "r") as f:
        for line in f.readlines():
            dic = json.loads(line)
            res.append(dic)

    value1 = []
    value2 = []
    absciss = []
    for i in range(len(res)):
        if key not in res[i]:
            continue
        if type(res[i][key]) != list:
            res[i][key] = [res[i][key]]

        me = np.array(res[i][key])
        value1.append(me.min())
        value2.append(me.max())
        absciss.append(i)
    return np.array(absciss), np.array(value1), np.array(value2)


# Stats bundle creation  -----------------------------------------------------------------------------------------------
def bundle_stats(agents, envs):
    saved_stats = {
        "Dist_Mean": mean_ag_dist,
        # '2D_benchmark': benchmark_evolution
    }

    dic = dict()
    for key in saved_stats.keys():
        dic[key] = saved_stats[key](agents, envs)

    if len(agents) == len(envs):
        dic["Paired_fitness"] = paired_fitness(agents, envs)
    return dic


def append_stats(path, bundle):
    with open(path, "a") as f:
        f.write(json.dumps(bundle))
        f.write("\n")
