import numpy as np
from Parameters import Configuration


def ES_Step(theta, E, args):
    """Local optimization by Evolution Strategy steps, rank normalization and weight decay"""
    og_weights = theta.get_weights()

    shared_gaussian_table = [np.random.normal(0, 1, size=len(og_weights)) for i in range(args.batch_size)]

    thetas = []
    for i in range(args.batch_size):
        new_theta = Configuration.agentFactory.new()
        new_theta.set_weights(og_weights + args.sigma * shared_gaussian_table[i])
        thetas.append(new_theta)

    scores = Configuration.lview.map(E, thetas)
    scores = np.array(scores)
    for i in range(len(scores)):
        scores[i] -= args.l_decay * np.linalg.norm(og_weights + args.sigma * shared_gaussian_table[i])

    if args.verbose > 0:
        print("\n\t Local best score :", scores.max())
    print("\n\t")
    scores = rank_normalize(scores)
    Configuration.budget_spent[-1] += len(thetas)
    
    summed_weights = np.zeros(og_weights.shape)
    for i in range(len(scores)):
        summed_weights += scores[i] * shared_gaussian_table[i]
    new_weights = (args.alpha/(len(shared_gaussian_table)*args.sigma)) * summed_weights
    new_weights += og_weights

    new_ag = Configuration.agentFactory.new()
    new_ag.set_weights(new_weights)
    return new_ag


def rank_normalize(arr):
    asorted = arr.argsort()
    linsp = np.linspace(0, 1, num=len(asorted))
    res = np.zeros(len(asorted))
    for i in range(len(asorted)):
        res[asorted[i]] = linsp[i]
    return res
