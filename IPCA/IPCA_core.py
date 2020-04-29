import numpy as np
from Parameters import Configuration


def non_dominated(elements):
    res = list()
    for i in range(len(elements)):
        dominated = False
        for j in range(len(elements)):
            if dominates(elements[i], elements[j]):
                dominated = True
                break
        if not dominated:
            res.append(elements[i])
    return res


def useful(element, comparison_list):
    """element is expected to be a list of boolean - binary tests results.
     The function then returns a boolean indicating if dominated by at least one element in the comparison list."""
    for comparison in comparison_list:
        if dominates(comparison, element):
            return False
    return True


def xuseful(element, comparison_list):
    """element is expected to be a list of boolean - binary tests results.
     The function then returns a boolean indicating if xdominated by at least one element in the comparison list."""
    for comparison in comparison_list:
        if xdominates(comparison, element):
            return False
    return True


def dominates(a, b):
    """Return true if a strictly dominate b"""
    for i in range(len(a)):
        if not a[i] and b[i]:
            return False
    return True


def xdominates(a, b):
    """Return true if a strictly dominate b, except when a is all true"""
    all_true = True
    for i in range(len(a)):
        if not a[i] and b[i]:
            return False
        all_true = all_true and a[i]
    if all_true:
        return False
    return True


def generate_learners(old_learners, args):
    # Mutation, reproduction & new ones
    Configuration.agentFactory.new()
    for i in range(args.Pop_size):
        pass
    pass


def generate_tests(old_tests):
    # Mutation, reproduction & new ones
    pass


def cross_evaluation(learners, tests, args):
    l_res = [[False for i in range(len(tests))] for j in range(len(learners))]
    t_res = [[False for i in range(len(learners))] for j in range(len(tests))]
    for i in range(len(tests)):
        res = Configuration.lview.map(tests[i], learners)
        ranks = np.array(res).argsort()[::-1]
        for j in ranks[:args.nb_best]:
            l_res[j][i] = True
            t_res[i][j] = True
    return l_res, t_res
