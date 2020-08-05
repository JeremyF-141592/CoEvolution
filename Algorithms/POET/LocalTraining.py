import numpy as np
from Algorithms.NSGA2.NSGAII_tools import add_objectives, obj_mean_observation_novelty, obj_mean_fitness
from Algorithms.NSGA2.NSGAII_core import fast_non_dominated_sort, crowding_distance
from Parameters import Configuration


def ES_Step(theta, E, args, allow_verbose=0):
    """Local optimization by Evolution Strategy steps, rank normalization and weight decay."""
    og_weights = theta.get_weights()

    shared_gaussian_table = [np.random.normal(0, 1, size=len(og_weights)) for i in range(args.batch_size)]

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

    step, new_state = Configuration.optimizer.step(grad_estimate, theta.get_opt_state(), args)

    new_ag = Configuration.agentFactory.new()
    new_ag.set_opt_state(new_state)
    new_ag.set_weights(og_weights + step)
    return new_ag, self_fitness


def NSGAII_step(theta, E, args, allow_verbose=0):
    """Local optimization by Evolution Strategy steps, rank normalization and weight decay."""

    args.gen_size = args.batch_size
    og_weights = theta.get_weights()

    shared_gaussian_table = [np.random.normal(0, 1, size=len(og_weights)) for i in range(args.batch_size)]

    sigma = max(args.noise_limit, args.noise_std * args.noise_decay ** theta.get_opt_state()["t"])

    thetas = []
    for i in range(args.batch_size):
        new_theta = Configuration.agentFactory.new()
        new_theta.set_weights(og_weights + sigma * shared_gaussian_table[i])
        thetas.append(new_theta)

    new_pop, objs = NSGAII_without_generation(thetas, [E], [obj_mean_fitness, obj_mean_observation_novelty], args)

    fit_list = list()
    for tup in objs:
        fit_list.append(tup[0])
    fit_list = np.array(fit_list)
    best_ag = new_pop[fit_list.argmax()]

    if allow_verbose > 0 and args.verbose > 0:
        print(f"Fitness : {round(fit_list.max(), 2)}", end="", flush=True)

    return best_ag, fit_list.max()


def NSGAII_without_generation(new_pop, envs, additional_objectives, args):
    # Both fitness and observation are required
    fit = [list() for i in range(len(envs))]
    obs = [list() for i in range(len(envs))]
    for i in range(len(envs)):
        res = Configuration.lview.map(envs[i], new_pop)
        Configuration.budget_spent[-1] += len(res)
        if type(res[0]) != tuple and type(res[0]) != list:
            raise TypeError("Current fitness metric returns a scalar instead of a tuple.")
        for r in res:
            fit[i].append(r[0])
            obs[i].append(r[1])

    results = add_objectives(fit, obs, new_pop, additional_objectives, envs, args)

    nd_sort = np.array(fast_non_dominated_sort(results))

    fronts = [list() for i in range(nd_sort.max() + 1)]  # store agents
    fronts_objectives = [list() for i in range(nd_sort.max() + 1)]  # store agents objectives
    for i in range(len(new_pop)):
        fronts[nd_sort[i]].append(new_pop[i])
        fronts_objectives[nd_sort[i]].append(results[i])

    pop = list()  # New population
    objs = list()  # Corresponding objectives
    last_front = 0
    for i in range(len(fronts)):
        if len(pop) + len(fronts[i]) > args.pop_size:
            break
        pop = pop + fronts[i]
        objs = objs + fronts_objectives[i]
        last_front = i + 1

    if last_front < len(fronts):
        cdistance = np.array(crowding_distance(fronts_objectives[last_front]))
        cdist_sort = cdistance.argsort()[::-1]

        # print("Keep", last_front, "fronts and add", args.pop_size - len(pop), "individuals via crowding distance.")

        for i in range(args.pop_size - len(pop)):  # fill the population with less crowded individuals of the last front
            pop.append(fronts[last_front][cdist_sort[i]])
            objs.append(fronts_objectives[last_front][cdist_sort[i]])

    # Return new population and their objectives
    return pop, objs


def rank_normalize(arr):
    asorted = arr.argsort()
    linsp = np.linspace(0, 1, num=len(asorted))
    res = np.zeros(len(asorted))
    for i in range(len(asorted)):
        res[asorted[i]] = linsp[i]
    return 2*res - 1

# ---------------------------------------------------------------------------------------- Local algorithm specification

Local_Algorithm = NSGAII_step

# ----------------------------------------------------------------------------------------------------------------------
