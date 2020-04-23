import numpy as np
from Parameters import Configuration


def ES_Step(theta, E, args, verbose=0):
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
    if verbose > 0:
        print(f"\n\tMean score : {round(scores.mean(), 2)}   Max score : {round(scores.max(), 2)}", end="", flush=True)

    for i in range(len(scores)):
        scores[i] -= args.w_decay * np.linalg.norm(og_weights + args.sigma * shared_gaussian_table[i])

    scores = rank_normalize(scores)
    Configuration.budget_spent[-1] += len(thetas)

    summed_weights = np.zeros(og_weights.shape)
    for i in range(len(scores)):
        summed_weights += scores[i] * shared_gaussian_table[i]
    grad_estimate = (1/(len(shared_gaussian_table))) * summed_weights

    alpha = 1
    t = 0
    if len(theta.get_opt_state()) > 0:
        alpha = theta.get_opt_state()[0]
        t = theta.get_opt_state()[1]
    
    step = grad_estimate * alpha

    alpha = max(args.lr_decay**t, args.lr_limit)
    t += 1

    new_ag = Configuration.agentFactory.new()
    new_ag.set_opt_state([alpha, t])
    new_ag.set_weights(og_weights + step)
    return new_ag


def rank_normalize(arr):
    asorted = arr.argsort()
    linsp = np.linspace(0, 1, num=len(asorted))
    res = np.zeros(len(asorted))
    for i in range(len(asorted)):
        res[asorted[i]] = linsp[i]
    return 2*res - 1
