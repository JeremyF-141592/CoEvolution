import numpy as np
from Parameters import Configuration


def fast_non_dominated_sort(elements):
    """
    elements is expected to be an array (m, n) of n objectives for m agents.
    Returns ranks as a list.
    """
    S = [list() for i in range(len(elements))]
    n = [0 for i in range(len(elements))]
    f_i = list()

    ranks = [0 for i in range(len(elements))]

    for p in range(len(elements)):
        for q in range(len(elements)):
            if dominates(elements[p], elements[q]):
                S[p].append(q)
            elif dominates(elements[q], elements[p]):
                ranks[p] = 1
                n[p] += 1
        if n[p] == 0:
            f_i.append(p)

    i = 0
    f_i = f_i
    while len(f_i) != 0:
        Q = list()
        for p in f_i:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    ranks[q] = i + 1
                    Q.append(q)
        i += 1
        f_i = Q
    return ranks


def dominates(a, b):
    """Return true if a Pareto dominates b (maximization)"""
    equals = True
    for i in range(len(a)):
        equals = equals and a[i] == b[i]
        if a[i] < b[i]:
            return False
    if equals:
        return False
    return True


def crowding_distance(elements):
    """elements is expected to be an array (m, n) of n objectives for m agents"""
    distance = [0.0 for i in range(len(elements))]
    for i in range(len(elements[0])):
        sorted_elements = sorted(elements, key=lambda x: x[i])
        distance[0] = float("inf")
        distance[-1] = float("inf")
        maxi = max(sorted_elements, key=lambda x: x[i])
        mini = min(sorted_elements, key=lambda x: x[i])
        for k in range(1, len(elements) - 1):
            distance[k] += (sorted_elements[k + 1][i] - sorted_elements[k - 1][i]) / (maxi - mini)
    return distance


def new_population(old_learners, args):
    # Mutation, reproduction & new ones
    new_learners = list()
    for i in range(args.pop_size):
        action = np.random.random()
        if len(old_learners) == 0:
            action = 1  # Ensure new tests only if there are currently no tests
        if action < args.p_mut_ag:
            new_learners.append(mutate_ag(old_learners[np.random.randint(0, len(old_learners))], args))
        elif action < args.p_cross_ag:
            choices = np.random.choice(np.arange(len(old_learners)), 2)
            new_learners.append(mate_ag(old_learners[choices[0]], old_learners[choices[1]]))
        else:
            new = Configuration.agentFactory.new()
            new.randomize()
            new_learners.append(new)
    return new_learners


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
