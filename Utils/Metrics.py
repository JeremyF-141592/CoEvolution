import numpy as np


def fitness_metric(agent, fitness, observation):
    return fitness


def novelty_metric(archive):
    def novelty_score(agent, fitness, observation):
        return fitness
    return novelty_score
