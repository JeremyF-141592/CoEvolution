import numpy as np
from Environments.reproduce_ops import Reproducer
from Parameters import Configuration
from POET.Selection import Evaluate_Candidates

fitness_median = float("-inf")


def mutate_envs(ea_list, args):
    """Mutate environments as in the first POET implementation"""
    parent_list = list()
    ea_len = len(ea_list)
    for m in range(ea_len):
        if eligible_to_reproduce(ea_list[m]):
            parent_list.append(ea_list[m])

    child_list = env_reproduce(parent_list, args.max_children, args)
    child_list, scores = mc_satisfied(child_list, args)
    child_list = rank_by_score(child_list, scores)
    admitted = 0
    for E_child, theta_child in child_list:
        theta_child = Evaluate_Candidates(child_list, E_child, args)
        if mc_satisfied([(E_child, theta_child)], args):
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
    return True


def mc_satisfied(child_list, args):
    """Check that every pair has a batch score between a predifined min and max."""
    env_scores = list()
    new_list = list()
    results = np.zeros(len(child_list))
    for i in range(args.nb_rounds):
        partial_result = Configuration.lview.map(paired_execution, child_list)
        for k in range(len(partial_result)):
            results[k] += partial_result[k]
    results /= args.nb_rounds
    for i in range(len(results)):
        if args.mc_min < results[i] < args.mc_max:
            new_list.append(child_list[i])
            env_scores.append(results[i])
    return new_list, env_scores


def rank_by_score(child_list, scores):
    arg_sort = np.array(scores).argsort()
    child_list = [child_list[i] for i in arg_sort]
    return child_list


def env_reproduce(parent_list, max_children, args):
    """Reproduce envs as in the first POET implementation"""
    rep = Reproducer(args)
    new_list = list(parent_list)
    choices = np.random.choice(np.arange(len(parent_list)), max_children)
    for i in choices:
        E, theta = parent_list[i]
        new_list.append((Configuration.baseEnv(rep.mutate(E)), theta))
    return new_list


def paired_execution(ea_pair):
    E, theta = ea_pair
    return E(theta)

