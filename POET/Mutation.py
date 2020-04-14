import numpy as np
from Parameters import Configuration
from POET.Transfer import Evaluate_Candidates
import matplotlib.pyplot as plt


def mutate_envs(ea_list, args):
    """Mutate environments as in the first POET implementation"""
    parent_list = list()
    ea_len = len(ea_list)
    for m in range(ea_len):
        if eligible_to_reproduce(ea_list[m]):
            parent_list.append(ea_list[m])

    child_list = env_reproduce(parent_list, args.max_children)
    child_list, scores = mc_satisfied(child_list, args)
    child_list = rank_by_score(child_list, scores)
    admitted = 0
    for E_child, theta_child in child_list:
        theta_child = Evaluate_Candidates(child_list, E_child, args)
        # if mc_satisfied([(E_child, theta_child)], args):
        if mc_satisfied_theta(E_child, theta_child, args):
            ea_list.append((E_child, theta_child))
            admitted += 1
            if admitted >= args.max_admitted:
                break
    ea_len = len(ea_list)
    if ea_len > args.Pop_size:
        num_removals = ea_len-args.Pop_size
        ea_list = ea_list[num_removals:]
    return ea_list


def eligible_to_reproduce(ea_pair):
    # TODO - find something useful to put here, original implementation -> check for duplicates
    # Here, we just expand the environment archive
    E, theta = ea_pair
    env_vector = np.array(E.__getstate__()["as_vector"])
    for k in Configuration.archive:
        vec = np.array(k["as_vector"])
        if np.linalg.norm(vec - env_vector) < 1:
            return True
    Configuration.archive.append(E.__getstate__())

    return True


def mc_satisfied(child_list, args):
    """Check that every env has a novelty score in between a predefined min and max."""
    env_scores = list()
    new_list = list()
    results = np.zeros(len(child_list))

    full_env_list = list()
    theta_list = list()
    for ea_pair in child_list:
        E, theta = ea_pair
        full_env_list.append(E)
        theta_list.append(theta)
    for state in Configuration.archive:
        E = Configuration.baseEnv(Configuration.flatConfig)
        E.__setstate__(state)
        full_env_list.append(E)

    points = pata_ec(full_env_list, theta_list)

    for i in range(len(child_list)):
        # KNN Novelty score
        if len(Configuration.archive) == 0:
            results[i] = 0
        dist_list = np.zeros(len(Configuration.archive))
        env_vec = points[i]
        for j in range(len(Configuration.archive)):
            dist_list[j] += np.linalg.norm(env_vec - points[j + len(child_list)])

        dist_list.sort()
        results[i] = dist_list[:Configuration.knn].mean()
        plt.plot(child_list[i][0].terrain_y)
    plt.show()

    print("NOVELTY ENVS : ", results.max(), results.min(), len(results))
    for i in range(len(results)):
        if args.mc_min < results[i] < args.mc_max:
            new_list.append(child_list[i])
            env_scores.append(results[i])
    return new_list, env_scores


def rank_by_score(child_list, scores):
    arg_sort = np.array(scores).argsort()[::-1]
    child_list = [child_list[i] for i in arg_sort]
    return child_list


def env_reproduce(parent_list, max_children):
    """Make children envs"""
    new_list = list(parent_list)
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
        result = rank_normalize(result)
        res.append(result)
    return res


def rank_normalize(arr):
    asorted = arr.argsort()
    linsp = np.linspace(0, 1, num=len(asorted))
    res = np.zeros(len(asorted))
    for i in range(len(asorted)):
        res[asorted[i]] = linsp[i]
    return res - 0.5


def mc_satisfied_theta(E, theta, args):
    score = E(theta)
    return True if score > -50 else False
