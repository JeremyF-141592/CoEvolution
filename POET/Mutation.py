import numpy as np
from Parameters import Configuration
from POET.Transfer import Evaluate_Candidates


def mutate_envs(ea_list, args):
    """Mutate environments as in the first POET implementation"""
    parent_list = list()
    ea_len = len(ea_list)
    for m in range(ea_len):
        if eligible_to_reproduce(ea_list[m], args):
            parent_list.append(ea_list[m])
    if len(parent_list) == 0:
        print("No environments is eligible to reproduce.")
        return ea_list
    child_list = env_reproduce(parent_list, args.max_children)
    child_list = mc_satisfied(child_list, args)
    child_list = rank_by_score(child_list, args)
    admitted = 0
    for E_child, theta_child in child_list:
        theta_child, score_child = Evaluate_Candidates(child_list, E_child, args)
        if args.mc_max > score_child > args.mc_min:
            ea_list.append((E_child, theta_child))
            admitted += 1
            if admitted >= args.max_admitted:
                break

    ea_len = len(ea_list)
    if ea_len > args.capacity:
        num_removals = ea_len-args.capacity
        if args.verbose > 0:
            print(f"\n{num_removals} Environments replaced.")
        for k in range(num_removals):
            Configuration.archive.append(ea_list[k])
        ea_list = ea_list[num_removals:]
    else:
        if args.verbose > 0:
            print(f"\n{ea_len} Current active environments.")
    return ea_list


def eligible_to_reproduce(ea_pair, args):
    E, theta = ea_pair
    return E(theta) > args.repro_threshold


def mc_satisfied(child_list, args):
    """Check that every env pass the minimal criterion"""
    new_list = list()
    for ea_pair in child_list:
        E, theta = ea_pair
        if args.mc_max > E(theta) > args.mc_min:
            new_list.append(ea_pair)
    return new_list


def rank_by_score(child_list, args):
    if len(child_list) == 0:
        return child_list
    results = np.zeros(len(child_list))

    full_env_list = list()
    theta_list = list()
    for ea_pair in child_list:
        E, theta = ea_pair
        full_env_list.append(E)
        theta_list.append(theta)
    for ea_pair in Configuration.archive:
        E, theta = ea_pair
        full_env_list.append(E)
        theta_list.append(theta)

    points = pata_ec(full_env_list, theta_list)

    for i in range(len(child_list)):
        # KNN Novelty score
        if len(Configuration.archive) == 0:
            break
        dist_list = np.zeros(len(Configuration.archive))
        env_vec = points[i]
        for j in range(len(Configuration.archive)):
            dist_list[j] += np.linalg.norm(env_vec - points[j + len(child_list)])

        dist_list.sort()
        results[i] = dist_list[:args.knn].mean()
    if args.verbose > 0:
        print(f"\nMaximum children novelty : {round(results.max(), 2)} - Minimum : {round(results.min(), 2)}")
    arg_sort = results.argsort()[::-1]
    child_list = [child_list[i] for i in arg_sort]
    return child_list


def env_reproduce(parent_list, max_children):
    """Make children envs"""
    new_list = list()
    choices = np.random.choice(np.arange(len(parent_list)), max_children)
    for i in choices:
        E, theta = parent_list[i]
        new_list.append((E.get_child(), theta))
    return new_list


def paired_execution(ea_pair):
    E, theta = ea_pair
    return E(theta)


def pata_ec(envs, individuals):
    """Returns a list of vectors representing environments by ranking individuals on them."""
    res = list()
    for i in range(len(envs)):
        result = Configuration.lview.map(envs[i], individuals)
        result = np.array(result)
        result = normalize(result)
        res.append(result)
    return res


def normalize(arr):
    tol = 5
    res = np.zeros(len(arr))
    for i in range(len(arr)):
        res[i] = min(250, max(-50, arr[i]))//tol
    uniques = np.unique(res)
    uniques.sort()
    dic = dict()
    for i in range(len(uniques)):
        dic[uniques[i]] = i
    for i in range(len(arr)):
        res[i] = dic[res[i]]
    res /= res.max()
    return res
