from Algorithms.NSGA2.NSGAII_core import *
from Parameters import Configuration
from ABC.Environments import ParameterizedEnvironment
from scipy.spatial.distance import jensenshannon
from scipy.stats import norm
import numpy as np


def NSGAII(pop, envs, additional_objectives, args):

    new_pop = pop + new_population(pop, args)

    # Both fitness and observation are required
    fit = [list() for i in range(len(envs))]
    obs = [list() for i in range(len(envs))]
    for i in range(len(envs)):
        res = Configuration.lview.map(envs[i], new_pop)
        Configuration.budget_spent[-1] += len(res)
        if type(res[0]) != tuple and type(res[0]) != list:
            raise TypeError("Current fitness metric returns a scalar instead of a tuple.")
        for r in res:
            fit[i].append(r[0])
            obs[i].append(r[1])

    results = add_objectives(fit, obs, new_pop, additional_objectives, envs, args)

    nd_sort = np.array(fast_non_dominated_sort(results))

    fronts = [list() for i in range(nd_sort.max() + 1)]  # store agents
    fronts_objectives = [list() for i in range(nd_sort.max() + 1)]  # store agents objectives
    for i in range(len(new_pop)):
        fronts[nd_sort[i]].append(new_pop[i])
        fronts_objectives[nd_sort[i]].append(results[i])

    pop = list()  # New population
    objs = list()  # Corresponding objectives
    last_front = 0
    for i in range(len(fronts)):
        if len(pop) + len(fronts[i]) > args.pop_size:
            break
        pop = pop + fronts[i]
        objs = objs + fronts_objectives[i]
        last_front = i + 1

    if last_front < len(fronts):
        cdistance = np.array(crowding_distance(fronts_objectives[last_front]))
        cdist_sort = cdistance.argsort()[::-1]

        # print("Keep", last_front, "fronts and add", args.pop_size - len(pop), "individuals via crowding distance.")

        for i in range(args.pop_size - len(pop)):  # fill the population with less crowded individuals of the last front
            pop.append(fronts[last_front][cdist_sort[i]])
            objs.append(fronts_objectives[last_front][cdist_sort[i]])

    # Return new population and their objectives
    return pop, objs


def add_objectives(fitness, observation, new_pop, objectives, envs, args):
    new_res = [list() for i in range(len(new_pop))]
    for i in range(len(new_pop)):
        for j in range(len(objectives)):
            new_res[i].append(objectives[j](i, fitness, observation, new_pop, envs, args))
    return new_res


def add_env_objectives(fitness, observation, new_pop, objectives, envs, args):
    new_res = [list() for i in range(len(envs))]
    for i in range(len(envs)):
        for j in range(len(objectives)):
            new_res[i].append(objectives[j](i, fitness, observation, new_pop, envs, args))
    return new_res


# Levenshtein distance, useful to compute Collectball BC ---------------------------------------------------------------

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1] + 1,
                    matrix[x, y-1] + 1
                )
    return matrix[size_x - 1, size_y - 1]
# ----------------------------------------------------------------------------------------------------------------------


# Objectives -----------------------------------------------------------------------------------------------------------
def obj_genotypic_novelty(index, fitness, observation, new_pop, envs, args):
    w = np.array(new_pop[index].get_weights())
    dists = np.zeros(len(new_pop))
    for j in range(len(new_pop)):
        dists[j] = np.linalg.norm(w - np.array(new_pop[j].get_weights()))
    dists.sort()
    return dists[:args.knn].mean()


def obj_mean_observation_novelty(index, fitness, observations, new_pop, envs, args):
    """Return k-nn novelty over observations, the mean is relative to observations on different environments."""
    res = 0
    for observation in observations:
        w = np.array(observation[index])
        dists = np.zeros(len(new_pop))
        for j in range(len(new_pop)):
            dists[j] = np.linalg.norm(w - np.array(observation[j]))
        dists.sort()
        res += dists[:args.knn].mean()
    return res / len(observations)


def obj_levenshtein_novelty(index, fitness, observations, new_pop, envs, args):
    """Return Levenshtein distance of discrete observations."""
    res = 0
    for observation in observations:
        w = np.array(observation[index]) // 50
        dists = np.zeros(len(new_pop))
        for j in range(len(new_pop)):
            dists[j] = levenshtein(w, np.array(observation[j]) // 50)
        dists.sort()
        res += dists[:args.knn].mean()
    return res / len(observations)


def obj_generalisation(index, fitness, observation, new_pop, envs, args):
    # p-mean over environments fitness
    values = list()
    for i in range(len(envs)):
        values.append(fitness[i][index])
    values = np.array(values)
    values.sort()
    dot = np.arange(len(values)) + 1
    dot = np.power(dot, args.mean)
    dot = dot / dot.sum()
    return np.multiply(values, dot).sum()


def obj_generalist_novelty(index, fitness, observation, new_pop, envs, args):
    # todo : replace with true generalist novelty, such as inverted pata-ec
    return obj_mean_observation_novelty(index, fitness, observation, new_pop, envs, args)


def obj_mean_fitness(index, fitness, observation, new_pop, envs, args):
    """Return mean fitness over environments."""
    res = 0
    for i in range(len(envs)):
        res += fitness[i][index]
    return res / len(fitness)


def obj_env_pata_ec(index, fitness, observation, new_pop, envs, args):
    w = normalize_pata_ec(fitness[index], args)  # list of every agent fitness on environment indexed at 'index'
    dists = np.zeros(len(envs))
    for j in range(len(envs)):
        dists[j] = np.linalg.norm(w - normalize_pata_ec(fitness[index], args))
    dists.sort()
    return dists[:args.knn_env].mean()


def obj_env_forwarding(index, fitness, observation, new_pop, envs, args):
    ev = envs[index]

    if "past_score" in dir(ev):
        new_max = np.array(fitness[index]).max()
        val = new_max - ev.past_score
        ev.past_score = new_max
        return val
    else:
        ev.past_score = np.array(fitness[index]).max()
        return 0


def obj_parametrized_env_novelty(index, fitness, observation, new_pop, envs, args):
    if not isinstance(envs[0], ParameterizedEnvironment):
        return 0
    w = np.array(envs[index].get_weights())
    dists = np.zeros(len(envs))
    for j in range(len(envs)):
        dists[j] = np.linalg.norm(w - np.array(envs[j].get_weights()))
    dists.sort()
    return dists[:args.knn].mean()


def obj_jensen_shannon(index, fitness, observation, new_pop, envs, args):
    bins = [i for i in range(-5, 5)]

    fit_distribution = list()
    for i in range(len(envs)):
        fit_distribution.append(fitness[i][index])
    fit_distribution = np.array(fit_distribution)

    if len(fit_distribution) <= 1:
        return 0

    esp = fit_distribution.mean()
    std = fit_distribution.std()
    fit_distribution = (fit_distribution - esp) / (std + 1e-8)

    fit_distribution_de = np.histogram(fit_distribution, bins=bins, density=True)[0]

    normal_bins = [norm.cdf(bins[i + 1]) - norm.cdf(bins[i]) for i in range(len(bins) - 1)]

    return -jensenshannon(fit_distribution_de, normal_bins, base=2)


# Generate Environments ------------------------------------------------------------------------------------------------
def generate_environments(envs, args):
    """Generate new environments by mutating old environments"""
    new_list = list()
    if len(envs) == 0:
        new_list.append(Configuration.envFactory.new())
        return new_list
    for i in range(args.max_env_children):
        choice = np.random.randint(0, len(envs))
        new_list.append(envs[choice].get_child())
    return new_list


def NSGAII_env(pop, envs, additional_objectives, args):
    """Applies evaluation, non-dominated sort and crowding sort to the environments."""
    # Both fitness and observation are required
    fit = [list() for i in range(len(envs))]
    obs = [list() for i in range(len(envs))]
    for i in range(len(envs)):
        res = Configuration.lview.map(envs[i], pop)
        Configuration.budget_spent[-1] += len(res)
        if type(res[0]) != tuple and type(res[0]) != list:
            raise TypeError("Current fitness metric returns a scalar instead of a tuple.")
        for r in res:
            fit[i].append(r[0])
            obs[i].append(r[1])

    results = add_env_objectives(fit, obs, pop, additional_objectives, envs, args)

    nd_sort = np.array(fast_non_dominated_sort(results))

    fronts = [list() for i in range(nd_sort.max() + 1)]  # store envs
    fronts_objectives = [list() for i in range(nd_sort.max() + 1)]  # store envs objectives

    for i in range(len(envs)):
        fronts[nd_sort[i]].append(envs[i])
        fronts_objectives[nd_sort[i]].append(results[i])

    pop = list()  # New population
    last_front = 0
    for i in range(len(fronts)):
        if len(pop) + len(fronts[i]) > args.pop_env_size:
            break
        pop = pop + fronts[i]
        last_front = i + 1

    if last_front < len(fronts):
        cdistance = np.array(crowding_distance(fronts_objectives[last_front]))
        cdist_sort = cdistance.argsort()[::-1]

        for i in range(args.pop_env_size - len(pop)):  # fill the env population with less crowded individuals of the last front
            pop.append(fronts[last_front][cdist_sort[i]])

    # Return new env population
    return pop


def normalize_pata_ec(arr, args):
    tol = args.pata_ec_tol
    res = np.zeros(len(arr))
    for i in range(len(arr)):
        if abs(tol) > 1e-4:
            res[i] = min(args.pata_ec_clipmax, max(args.pata_ec_clipmin, arr[i]))//tol
        else:
            res[i] = min(args.pata_ec_clipmax, max(args.pata_ec_clipmin, arr[i]))
    uniques = np.unique(res)
    uniques.sort()
    dic = dict()
    for i in range(len(uniques)):
        dic[uniques[i]] = len(uniques) - i
    for i in range(len(arr)):
        res[i] = dic[res[i]]
    res /= res.max()
    return res


def bundle_stats_NNSGA(local_bool, objs_local, objs_general, args):
    bundle = dict()
    if local_bool:
        for k in range(len(objs_local[0][0])):
            bundle[f"Objective_{k}-max"] = list()
        for i in range(len(objs_local)):
            for k in range(len(objs_local[i][0])):  # reformat objectives from list of tuple to lists for each objective
                obj_list = list()
                for j in range(len(objs_local[i])):
                    obj_list.append(objs_local[i][j][k])
                obj_arr = np.array(obj_list)
                bundle[f"Objective_{k}-max"].append(obj_arr.max())
    else:
        for k in range(len(objs_general[0])):  # reformat objectives from list of tuple to lists for each objective
            bundle[f"Objective_general-{k}"] = list()
            maxi = float("-inf")
            for j in range(len(objs_general)):
                bundle[f"Objective_general-{k}"].append(objs_general[j][k])
                if objs_general[j][k] > maxi:
                    maxi = objs_general[j][k]

    return bundle
