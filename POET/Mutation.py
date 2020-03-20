import numpy as np
from Environments.reproduce_ops import Reproducer
from Parameters import baseEnv
from POET.Selection import Evaluate_Candidates

fitness_median = float("-inf")


def Mutate_Envs(EA_list, args):
    parent_list = list()
    M = len(EA_list)
    for m in range(M):
        if eligible_to_reproduce(EA_list[m]):
            parent_list.append(EA_list[m])

    child_list = env_reproduce(parent_list, args.max_children, args)
    child_list, scores = mc_satisfied(child_list, args)
    child_list = rank_by_score(child_list, scores)
    admitted = 0
    for E_child, theta_child in child_list:
        theta_child = Evaluate_Candidates(child_list, E_child, args)
        if mc_satisfied([(E_child, theta_child)], args):
            EA_list.append((E_child, theta_child))
            admitted += 1
            if admitted >= args.max_admitted:
                break
    M = len(EA_list)
    if M > args.capacity:
        num_removals = M-args.capacity
        EA_list = EA_list[num_removals:]
    return EA_list


def eligible_to_reproduce(ea_pair):
    # TODO - find something useful to put here, original implementation -> check for duplicates
    return True


def mc_satisfied(child_list, args):
    """Check that every pair has a batch score at least better than the previous median batch score."""
    env_scores = list()
    new_list = list()
    for i in range(len(child_list)):
        E, theta = child_list[i]
        mean = 0
        for j in range(args.nb_rounds):
            mean += E(theta)
        mean /= args.nb_rounds
        if mean > fitness_median:
            new_list.append(child_list[i])
            env_scores.append(mean)
    return new_list, env_scores


def rank_by_score(child_list, scores):
    arg_sort = np.array(scores).argsort()
    child_list = child_list[arg_sort]
    return child_list


def env_reproduce(parent_list, max_children, args):
    rep = Reproducer(args)
    new_list = list(parent_list)
    choices = np.random.choice(np.arange(len(parent_list)), max_children)
    for i in choices:
        E, theta = parent_list[i]
        new_list.append(baseEnv(rep.mutate(E)))
    return new_list



