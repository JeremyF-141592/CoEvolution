# A metric is a function that takes in (agent, environment, raw_fitness, path)
# and returns a tuple (fitness, observation), or just the fitness depending on what is needed.
# 'path' denotes the list of observation the agent went through.

import numpy as np


def fitness_metric(agent, environment, raw_fitness, path):
    """ Returns the raw fitness of an agent evaluation. """
    return raw_fitness


def fitness_bc(agent, environment, raw_fitness, path):
    """ Returns as a tuple the raw fitness and raw observation of an agent evaluation. """
    if len(path) > 0:
        return raw_fitness, path
    return raw_fitness, [0]
