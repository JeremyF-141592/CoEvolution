import numpy as np
from Parameters import Configuration
from Algorithms.POET.Transfer import Evaluate_Candidates


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

    if len(child_list) == 0 and args.verbose > 0:
        print("No child environments passed the Minmal Criterion.")
    elif args.verbose > 0:
        print(f"Environments that passed MC : {len(child_list)} / {args.max_children}")

    original_thetas = list()  # extracting parent agents
    for ea_pair in parent_list:
        E, theta = ea_pair
        original_thetas.append(theta)

    child_list = rank_by_score(child_list, original_thetas, args)
    admitted = 0
    for E_child, theta_child in child_list:
        theta_child, score_child = Evaluate_Candidates(parent_list, E_child, args, threshold=args.mc_min)
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
    if args.verbose > 0:
        print(f"\n{len(ea_list)} Current active environments.")
    return ea_list


def eligible_to_reproduce(ea_pair, args):
    E, theta = ea_pair
    sc = E(theta)
    if type(sc) == tuple or type(sc) == list:
        sc = sc[0]
    return sc > args.repro_threshold


def mc_satisfied(child_list, args):
    """Check that every env pass the minimal criterion"""
    new_list = list()
    for ea_pair in child_list:
        E, theta = ea_pair
        if args.mc_max > E(theta) > args.mc_min:
            new_list.append(ea_pair)
    return new_list


def rank_by_score(child_list, original_thetas, args):
    if len(child_list) == 0:
        return child_list
    results = np.zeros(len(child_list))

    full_env_list = list()
    theta_list = original_thetas
    for ea_pair in child_list:
        E, theta = ea_pair
        full_env_list.append(E)
    for ea_pair in Configuration.archive:
        E, theta = ea_pair
        full_env_list.append(E)
        theta_list.append(theta)

    points = pata_ec(full_env_list, theta_list, args)

    for i in range(len(child_list)):
        # KNN Novelty score
        if len(Configuration.archive) == 0:
            break
        dist_list = np.zeros(len(full_env_list))
        env_vec = points[i]
        for j in range(len(full_env_list)):
            dist_list[j] += np.linalg.norm(env_vec - points[j])

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


def pata_ec(envs, individuals, args):
    """Returns a list of vectors representing environments by ranking individuals on them."""
    res = list()
    for i in range(len(envs)):
        result = Configuration.lview.map(envs[i], individuals)
        Configuration.budget_spent[-1] += len(result)
        result = np.array(result)
        result = normalize(result, args)
        res.append(result)
    return res


def normalize(arr, args):
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
