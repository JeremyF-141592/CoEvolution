import numpy as np
import random
from Parameters import Configuration


def fast_non_dominated_sort(elements):
    """
    Fast non-dominated sort in the sens of Pareto, as in the original NSGA-II algorithm.
    :param elements: array (m, n) of n objectives for m agents.
    :return: front ranks as a list.
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
    distance = [0 for i in range(len(elements))]
    for i in range(len(elements[0])):
        sorted_elements = sorted(elements, key=lambda x: x[i])
        sorted_args = sorted(range(len(elements)), key=lambda x: elements[x][i])
        distance[sorted_args[0]] = float("inf")
        distance[sorted_args[-1]] = float("inf")
        maxi = sorted_elements[-1][i]
        mini = sorted_elements[0][i]
        for k in range(1, len(elements) - 1):
            distance[sorted_args[k]] += (sorted_elements[k + 1][i] - sorted_elements[k - 1][i]) / (maxi - mini + 1e-6)
    return distance


def new_population(old_learners, args):
    """
    Create a new agent population from a parent one.
    If the parent population does not contain any agent, returns a new random one.
    The strategy is mutation with probability p1, crossover with probability p2, random agent with probability 1-p1-p2.
    Mutants and crossover parents are chosen with equal probability from the parent list.

    :param old_learners: parent agent population
    :param args: argparse object
    :return: agent population as a list
    """
    # Mutation, reproduction & new ones
    new_learners = list()
    for i in range(args.gen_size):
        action = np.random.random()
        if len(old_learners) == 0:
            action = 1  # Ensure new agents only if there are currently no agents
        if action < args.p_mut_ag:
            new_learners.append(mutate_ag(old_learners[np.random.randint(0, len(old_learners))], args))
        elif action < args.p_cross_ag:
            choices = np.random.choice(np.arange(len(old_learners)), 2)
            new_learners.append(mate_ag(old_learners[choices[0]], old_learners[choices[1]], args))
        else:
            new = Configuration.agentFactory.new()
            new.randomize()
            new_learners.append(new)
    return new_learners


def mutate_ag(agent, args):
    return mutPolynomialBounded(agent, args.eta_mut, args.mut_low_bound, args.mut_high_bound, args.p_mut_gene)


def mate_ag(agent1, agent2, args):
    return cxSimulatedBinaryBounded(agent1, agent2, args.eta_cross, args.mut_low_bound, args.mut_high_bound)


def mutPolynomialBounded(agent, eta, low, up, indpb):
    """
    Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb, returns a new agent.
    This function is in parts taken from the DEAP package for python.
    """
    weights = agent.get_weights().copy()
    size = len(weights)
    xl = low
    xu = up
    for i in range(size):
        if random.random() <= indpb:
            x = weights[i]
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (xu - xl)
            x = min(max(x, xl), xu)
            weights[i] = x
    new = Configuration.agentFactory.new()
    new.set_weights(weights)
    return new


def cxSimulatedBinaryBounded(ind1, ind2, eta, xl, xu):
    """
    Executes a simulated binary crossover, returns a new agent.
    This function is in parts taken from the DEAP package for python.
    """
    w1 = ind1.get_weights().copy()
    w2 = ind2.get_weights().copy()
    for i in range(len(w1)):
        c1 = 0
        c2 = 0
        # This epsilon should probably be changed for 0 since
        # floating point arithmetic in Python is safer
        if abs(w1[i] - w2[i]) > 1e-14:
            x1 = min(w1[i], w2[i])
            x2 = max(w1[i], w2[i])
            rand = random.random()

            beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

            c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

            beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
            c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

            c1 = min(max(c1, xl), xu)
            c2 = min(max(c2, xl), xu)

        if random.random() <= 0.5:
            w1[i] = c2
        else:
            w1[i] = c1
    new = Configuration.agentFactory.new()
    new.set_weights(w1)
    return new
