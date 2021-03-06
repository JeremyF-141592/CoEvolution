import numpy as np
from Algorithms.NSGA2.NSGAII_tools import add_objectives, obj_mean_observation_novelty, obj_mean_fitness
from Algorithms.NSGA2.NSGAII_core import fast_non_dominated_sort, crowding_distance
from Parameters import Configuration


def ES_Step(theta, E, args, allow_verbose=0):
    """Local optimization by Evolution Strategy steps, rank normalization and weight decay."""
    og_weights = theta.get_weights()

    shared_gaussian_table = [np.random.normal(0, 1, size=len(og_weights)) for i in range(args.batch_size)]

    if theta.get_opt_state() is None:
        theta.set_opt_state(Configuration.optimizer.default_state())
    if "t" not in theta.get_opt_state().keys():
        z = theta.get_opt_state().copy()
        z.update({"t": 1})
        theta.set_opt_state(z)

    sigma = max(args.noise_limit, args.noise_std * args.noise_decay ** theta.get_opt_state()["t"])

    thetas = []
    for i in range(args.batch_size):
        new_theta = Configuration.agentFactory.new()
        new_theta.set_weights(og_weights + sigma * shared_gaussian_table[i])
        thetas.append(new_theta)

    scores = Configuration.lview.map(E, thetas)
    Configuration.budget_spent[-1] += len(thetas)
    scores = np.array(scores)

    self_fitness = E(theta)
    if allow_verbose > 0 and args.verbose > 0:
        print(f"Fitness : {round(self_fitness, 2)}   Mean batch fitness : {round(scores.mean(), 2)}",
              end="", flush=True)

    for i in range(len(scores)):
        scores[i] -= args.w_decay * np.linalg.norm(og_weights + sigma * shared_gaussian_table[i])

    scores = rank_normalize(scores)

    summed_weights = np.zeros(og_weights.shape)
    for i in range(len(scores)):
        summed_weights += scores[i] * shared_gaussian_table[i]
    grad_estimate = -(1/(len(shared_gaussian_table))) * summed_weights

    step, new_state = Configuration.optimizer.step(grad_estimate, theta.get_opt_state())

    new_ag = Configuration.agentFactory.new()
    new_ag.set_opt_state(new_state)
    new_ag.set_weights(og_weights + step)
    return new_ag, self_fitness


def rank_normalize(arr):
    asorted = arr.argsort()
    linsp = np.linspace(0, 1, num=len(asorted))
    res = np.zeros(len(asorted))
    for i in range(len(asorted)):
        res[asorted[i]] = linsp[i]
    return 2*res - 1
