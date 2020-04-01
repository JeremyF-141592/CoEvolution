import numpy as np
from Parameters import Configuration


def fitness_metric(agent, environment, fitness, observation, archive):
    return fitness


def environment_novelty_metric(agent, environment, fitness, observation, archive):
	# brut force knn, could be improved (KD Tree)
	if len(archive) == 0:
		return 0
	dist_list = np.zeros(len(archive));
	for i in range(len(archive)):
		dist_list.append(np.linalg.norm(archive[i] - environment))
	dist_list.sort()
	return dist_list[:Configuration.knn].mean()
