import numpy as np
from Parameters import Configuration


def fitness_metric(agent, environment, fitness, observation):
    return fitness


def nsga_fitness_bc(agent, environment, fitness, observation):
    dist = 0
    if len(Configuration.archive) > 0:
        choices = np.random.choice(np.arange(len(Configuration.archive)), size=10)
        for i in choices:
            c = np.array(Configuration.archive[i])
            dist += np.linalg.norm(observation - c)
        dist /= 10
    return fitness, dist
