import numpy as np


def fitness_metric(agent, environment, fitness, path):
    return fitness


def fitness_bc(agent, environment, fitness, path):
    p = np.array(path).T
    obs = p.mean(axis=1) + np.diag(np.cov(p))
    return fitness, obs
