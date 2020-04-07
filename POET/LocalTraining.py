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
    if args.verbose > 0:
        print("\n\t Local best score :", scores.max())

    for i in range(len(scores)):
        scores[i] -= args.w_decay * np.linalg.norm(og_weights + args.sigma * shared_gaussian_table[i])

    scores = rank_normalize(scores)
    Configuration.budget_spent[-1] += len(thetas)

    summed_weights = np.zeros(og_weights.shape)
    for i in range(len(scores)):
        summed_weights += scores[i] * shared_gaussian_table[i]
    grad_estimate = (args.alpha/(len(shared_gaussian_table)*args.sigma)) * summed_weights

    step, state = Adam.step(og_weights, grad_estimate, theta.get_opt_state(), args)

    new_ag = Configuration.agentFactory.new()
    new_ag.set_opt_state(state)
    new_ag.set_weights(og_weights + step)
    return new_ag


def rank_normalize(arr):
    asorted = arr.argsort()
    linsp = np.linspace(0, 1, num=len(asorted))
    res = np.zeros(len(asorted))
    for i in range(len(asorted)):
        res[asorted[i]] = linsp[i]
    return res


class Adam:
    @staticmethod
    def init_state(size):
        t = 1
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-08
        m = [0 for i in range(size)]
        v = [0 for i in range(size)]
        return t, beta1, beta2, epsilon, m, v

    @staticmethod
    def step(point, gradient, state, args):
        if len(state) == 0:
            state = Adam.init_state(len(point))
        t, beta1, beta2, epsilon, m, v = state
        m = np.array(m)
        v = np.array(v)
        stepsize = max(args.init_step * args.lr_decay**t, args.lr_limit)
        a = stepsize * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient * gradient)
        step = -a * m / (np.sqrt(v) + epsilon)
        t += 1
        newstate = (t, beta1, beta2, epsilon, m.tolist(), v.tolist())
        return step, newstate
