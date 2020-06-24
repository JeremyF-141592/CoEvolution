# A metric is a function that takes in (agent, environment, raw_fitness, path)
# and returns a tuple (fitness, observation).
# 'path' denotes the list of states the agent went through

import numpy as np


def fitness_metric(agent, environment, raw_fitness, path):
    return raw_fitness, [0]


def fitness_bc(agent, environment, raw_fitness, path):
    # p = np.array(path).T
    # obs = p.mean(axis=1) + np.diag(np.cov(p))
    return raw_fitness, [0]
