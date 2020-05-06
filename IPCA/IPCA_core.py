import numpy as np
from Parameters import Configuration


def non_dominated(elements, scores):
    if len(elements) == 0:
        return elements
    assert(len(elements) == len(scores))
    res = list()
    for i in range(len(elements)):
        dominated = False
        for j in range(len(elements)):
            if i != j and dominates(scores[j], scores[i]):
                dominated = True
                break
        if not dominated:
            res.append(elements[i])
    return res


def useful(elements, scores, comparison_list):
    """One score is expected to be a list of boolean - binary tests results.
     The function then returns elements indicating if its score is
     dominated by at least one element in the comparison list."""
    if len(comparison_list) == 0:
        return elements, [i for i in range(len(elements))]

    new_elements = list()
    new_scores = list()
    for i in range(len(scores)):
        is_dominated = False
        for comparison in comparison_list:
            if dominates(comparison, scores[i]):
                is_dominated = True
                break
        if not is_dominated and not scores[i] in comparison_list:
            new_scores.append(i)
            new_elements.append(elements[i])
    return new_elements, new_scores


def xuseful(elements, scores, comparison_list):
    """One score is expected to be a list of boolean - binary tests results.
     The function then returns elements indicating if its score is
     xdominated by at least one element in the comparison list."""
    if len(comparison_list) == 0:
        return elements, [i for i in range(len(elements))]

    new_elements = list()
    new_scores = list()
    for i in range(len(scores)):
        is_xdominated = False
        for comparison in comparison_list:
            if xdominates(comparison, scores[i]):
                is_xdominated = True
                break
        if not is_xdominated and not scores[i] in comparison_list:
            new_scores.append(i)
            new_elements.append(elements[i])
    return new_elements, new_scores


def dominates(a, b):
    """Return true if a strictly dominate b"""
    equals = True
    for i in range(len(a)):
        equals = equals and a[i] == b[i]
        if not a[i] and b[i]:
            return False
    if equals:
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
    new_learners = list()
    for i in range(args.new_ags):
        action = np.random.random()
        if len(old_learners) == 0:
            action = 1  # Ensure new tests only if there are currently no tests
        if action < args.p_mut_env:
            new_learners.append(mutate_ag(old_learners[np.random.randint(0, len(old_learners))], args))
        elif action < args.p_cross_env:
            choices = np.random.choice(np.arange(len(old_learners)), 2)
            new_learners.append(mate_ag(old_learners[choices[0]], old_learners[choices[1]]))
        else:
            new = Configuration.agentFactory.new()
            new.randomize()
            new_learners.append(new)
    return new_learners


def generate_tests(old_tests, args):
    new_tests = list()
    for i in range(args.new_tests):
        action = np.random.random()
        if len(old_tests) == 0:
            action = 1  # Ensure new tests only if there are currently no tests
        if action < args.p_mut_env:
            new_tests.append(old_tests[np.random.randint(0, len(old_tests))].get_child())
        elif action < args.p_cross_env:
            choices = np.random.choice(np.arange(len(old_tests)), 2)
            new_tests.append(old_tests[choices[0]].mate(old_tests[choices[1]]))
        else:
            new = Configuration.baseEnv(Configuration.flatConfig)
            for i in range(10):
                new = new.get_child()
            new_tests.append(new)
    return new_tests


def cross_evaluation(learners, tests, args):
    l_res = [[False for i in range(len(tests))] for j in range(len(learners))]
    t_res = [[False for i in range(len(learners))] for j in range(len(tests))]

    overall_max = -float("inf")

    for i in range(len(tests)):
        res = Configuration.lview.map(tests[i], learners)
        for j in res:
            if j > overall_max:
                overall_max = j
        ranks = np.array(res).argsort()[::-1]
        for j in ranks[:args.nb_best]:
            l_res[j][i] = True
            t_res[i][j] = True
    return l_res, t_res, overall_max


def mutate_ag(agent, args):
    new_wei = agent.get_weights()
    for i in range(len(new_wei)):
        if np.random.random() < args.p_mut_gene:
            new_wei[i] += np.random.uniform(-1, 1) * args.mut_step
    new = Configuration.agentFactory.new()
    new.set_weights(new_wei)
    return new


def mate_ag(agent1, agent2):
    wei_1 = agent1.get_weights()
    wei_2 = agent2.get_weights()
    new_wei = list()
    for i in range(len(wei_1)):
        if np.random.random() < 0.5:
            new_wei.append(wei_1[i])
        else:
            new_wei.append(wei_2[i])
    new = Configuration.agentFactory.new()
    new.set_weights(new_wei)
    return new
