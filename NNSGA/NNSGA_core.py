from Baseline.NSGAII_core import *
from Parameters import Configuration


def NSGAII(pop, envs, additional_objectives, args):

    new_pop = pop + new_population(pop, args)

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


def add_objectives(fitness, observation, new_pop, objectives, envs, args):
    new_res = [list() for i in range(len(new_pop))]
    for i in range(len(new_pop)):
        for j in range(len(objectives)):
            new_res[i].append(objectives[j](i, fitness, observation, new_pop, envs, args))
    return new_res


def add_env_objectives(fitness, observation, new_pop, objectives, envs, args):
    new_res = [list() for i in range(len(envs))]
    for i in range(len(envs)):
        for j in range(len(objectives)):
            new_res[i].append(objectives[j](i, fitness, observation, new_pop, envs, args))
    return new_res


# Objectives -----------------------------------------------------------------------------------------------------------
def obj_genotypic_novelty(index, fitness, observation, new_pop, envs, args):
    w = new_pop[index].get_weights()
    dists = np.zeros(len(new_pop))
    for j in range(len(new_pop)):
        dists[j] = np.linalg.norm(w - new_pop[j].get_weights())
    dists.sort()
    return dists[:args.knn].mean()


def obj_mean_observation_novelty(index, fitness, observations, new_pop, envs, args):
    """Return k-nn novelty over observations, the mean is relative to observations on different environments."""
    res = 0
    for observation in observations:
        w = np.array(observation[index])
        dists = np.zeros(len(new_pop))
        for j in range(len(new_pop)):
            dists[j] = np.linalg.norm(w - np.array(observation[j]))
        res += dists.sort()[:args.knn].mean()
    return res / len(observations)


def obj_generalisation(index, fitness, observation, new_pop, envs, args):
    # p-mean over environments fitness
    res = list()
    for i in range(len(envs)):
        res.append(fitness[i][index])
    return np.quantile(res, args.quantile)


def obj_generalist_novelty(index, fitness, observation, new_pop, envs, args):
    # todo : replace with true generalist novelty, such as inverted pata-ec
    return obj_genotypic_novelty(index, fitness, observation, new_pop, envs, args)


def obj_mean_fitness(index, fitness, observation, new_pop, envs, args):
    """Return mean fitness over environments."""
    res = 0
    for i in range(len(envs)):
        res += fitness[i][index]
    return res / len(fitness)


def obj_env_pata_ec(index, fitness, observation, new_pop, envs, args):
    w = normalize_pata_ec(fitness[index], args)  # list of every agent fitness on environment indexed at 'index'
    dists = np.zeros(len(envs))
    for j in range(len(envs)):
        dists[j] = np.linalg.norm(w - normalize_pata_ec(fitness[index], args))
    dists.sort()
    return dists[:args.knn_env].mean()


def obj_env_forwarding(index, fitness, observation, new_pop, envs, args):
    # todo : HoF forwarding
    return np.random.random()


# Generate Environments ------------------------------------------------------------------------------------------------
def generate_environments(envs, args):
    """Generate new environments by mutating old environments"""
    new_list = list()
    if len(envs) == 0:
        new_list.append(Configuration.envFactory.new())
        return new_list
    for i in range(args.max_env_children):
        choice = np.random.randint(0, len(envs))
        new_list.append(envs[choice].get_child())
    return new_list


def NSGAII_env(pop, envs, additional_objectives, args):
    """Applies evaluation, non-dominated sort and crowding sort to the environments."""
    # Both fitness and observation are required
    fit = [list() for i in range(len(envs))]
    obs = [list() for i in range(len(envs))]
    for i in range(len(envs)):
        res = Configuration.lview.map(envs[i], pop)
        Configuration.budget_spent[-1] += len(res)
        if type(res[0]) != tuple and type(res[0]) != list:
            raise TypeError("Current fitness metric returns a scalar instead of a tuple.")
        for r in res:
            fit[i].append(r[0])
            obs[i].append(r[1])

    results = add_env_objectives(fit, obs, pop, additional_objectives, envs, args)

    nd_sort = np.array(fast_non_dominated_sort(results))

    fronts = [list() for i in range(nd_sort.max() + 1)]  # store envs
    fronts_objectives = [list() for i in range(nd_sort.max() + 1)]  # store envs objectives

    for i in range(len(envs)):
        fronts[nd_sort[i]].append(envs[i])
        fronts_objectives[nd_sort[i]].append(results[i])

    pop = list()  # New population
    last_front = 0
    for i in range(len(fronts)):
        if len(pop) + len(fronts[i]) > args.pop_env_size:
            break
        pop = pop + fronts[i]
        last_front = i + 1

    if last_front < len(fronts):
        cdistance = np.array(crowding_distance(fronts_objectives[last_front]))
        cdist_sort = cdistance.argsort()[::-1]

        for i in range(args.pop_env_size - len(pop)):  # fill the env population with less crowded individuals of the last front
            pop.append(fronts[last_front][cdist_sort[i]])

    # Return new env population
    return pop


def normalize_pata_ec(arr, args):
    tol = args.pata_ec_tol
    res = np.zeros(len(arr))
    for i in range(len(arr)):
        if abs(tol) > 1e-4:
            res[i] = min(args.pata_ec_clipmax, max(args.pata_ec_clipmin, arr[i]))//tol
        else:
            res[i] = min(args.pata_ec_clipmax, max(args.pata_ec_clipmin, arr[i]))
    uniques = np.unique(res)
    uniques.sort()
    dic = dict()
    for i in range(len(uniques)):
        dic[uniques[i]] = len(uniques) - i
    for i in range(len(arr)):
        res[i] = dic[res[i]]
    res /= res.max()
    return res


def bundle_stats_NNSGA(local_bool, objs_local, objs_general, args):
    bundle = dict()
    if local_bool:
        for k in range(len(objs_local[0][0])):
            bundle[f"Objective_{k}-min"] = list()
            bundle[f"Objective_{k}-max"] = list()
            bundle[f"Objective_{k}-med"] = list()
            bundle[f"Objective_{k}-argmax"] = list()

        for i in range(len(objs_local)):
            for k in range(len(objs_local[i][0])):  # reformat objectives from list of tuple to lists for each objective
                obj_list = list()
                for j in range(len(objs_local[i])):
                    obj_list.append(objs_local[i][j][k])
                obj_arr = np.array(obj_list)
                bundle[f"Objective_{k}-min"].append(obj_arr.min())
                bundle[f"Objective_{k}-max"].append(obj_arr.max())
                bundle[f"Objective_{k}-argmax"].append(int(obj_arr.argmax()))
                bundle[f"Objective_{k}-med"].append(np.median(obj_arr))
    else:
        for k in range(len(objs_general[0])):  # reformat objectives from list of tuple to lists for each objective
            bundle[f"Objective_general-{k}"] = list()
            arg_maxi = 0
            maxi = float("-inf")
            for j in range(len(objs_general)):
                bundle[f"Objective_general-{k}"].append(objs_general[j][k])
                if objs_general[j][k] > maxi:
                    maxi = objs_general[j][k]
                    arg_maxi = j
            bundle[f"Objective_general-arg{k}"] = [int(arg_maxi)]

    return bundle
