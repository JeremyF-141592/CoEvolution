import numpy as np
from Parameters import Configuration
from POET.Transfer import Evaluate_Candidates


def mutate_envs(ea_list, args):
    """Mutate environments as in the first POET implementation"""
    parent_list = list()
    ea_len = len(ea_list)
    for m in range(ea_len):
        if eligible_to_reproduce(ea_list[m]):
            parent_list.append(ea_list[m])

    child_list = env_reproduce(parent_list, args.max_children)
    child_list = mc_satisfied(child_list, args)
    child_list = rank_by_score(child_list, args)
    admitted = 0
    for E_child, theta_child in child_list:
        theta_child = Evaluate_Candidates(child_list, E_child, args)
        if args.mc_max > E_child(theta_child) > args.mc_min:
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
        vec = np.array(k.__getstate__()["as_vector"])
        if np.linalg.norm(vec - env_vector) < 1:
            return True
    new_env = Configuration.baseEnv(Configuration.flatConfig)
    new_env.__setstate__(E.__getstate__())
    Configuration.archive.append(new_env)

    return True


def mc_satisfied(child_list, args):
    """Check that every env pass the minimal criterion"""
    new_list = list()
    for ea_pair in child_list:
        E, theta = ea_pair
        if args.mc_max > E(theta) > args.mc_min:
            new_list.append(ea_pair)
    return new_list


def rank_by_score(child_list, args):
    results = np.zeros(len(child_list))

    full_env_list = list()
    theta_list = list()
    for ea_pair in child_list:
        E, theta = ea_pair
        full_env_list.append(E)
        theta_list.append(theta)
    for env in Configuration.archive:
        full_env_list.append(env)

    points = pata_ec(full_env_list, theta_list)
    print(f"All points PATA-EC, archive starts at {len(child_list)} :")
    for i in range(len(points)):
        print(f"{i} : {points[i]}")

    for i in range(len(child_list)):
        # KNN Novelty score
        if len(Configuration.archive) == 0:
            results[i] = 0
        dist_list = np.zeros(len(Configuration.archive))
        env_vec = points[i]
        for j in range(len(Configuration.archive)):
            dist_list[j] += np.linalg.norm(env_vec - points[j + len(child_list)])

        dist_list.sort()
        print(f"Env {i}, knn dists : {dist_list[:args.knn]}")
        results[i] = dist_list[:args.knn].mean()
    print("NOVELTY ENVS : ", results.max(), results.min(), len(results))
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
    res = arr - arr.min()
    res /= res.max()
    return res
