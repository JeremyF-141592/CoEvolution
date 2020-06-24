# A metric is a function that takes in (agent, environment, raw_fitness, path)
# and returns a tuple (fitness, observation), or just the fitness depending on what is needed.
# 'path' denotes the list of states the agent went through

import numpy as np


def fitness_metric(agent, environment, raw_fitness, path):
    return raw_fitness


def fitness_bc(agent, environment, raw_fitness, path):
    # p = np.array(path).T
    # obs = p.mean(axis=1) + np.diag(np.cov(p))
    return raw_fitness, [0]
