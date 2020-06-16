from Baseline.NSGAII_core import *
from Parameters import Configuration


def nsga_iteration(pop, env, args):
    new_pop = new_population(pop, args)
    pop = list()

    results = Configuration.lview.map(env, new_pop)

    # GENOTYPIC NOVELTY ---- todo : parallelism ?
    for i in range(len(results)):
        w = new_pop[i].get_weights()
        dists = np.zeros(len(new_pop))
        for j in range(len(new_pop)):
            dists[j] = np.linalg.norm(w - new_pop[j].get_weights())
        results[i] = (results[i][0], dists.sort()[:args.knn].mean())

    nd_sort = np.array(fast_non_dominated_sort(results))

    fronts = [list() for i in range(nd_sort.max() + 1)]

    for i in range(len(new_pop)):
        fronts[nd_sort[i]].append(new_pop[i])

    for i in range(len(fronts)):
        if len(pop) > args.pop_size:
            break
        pop = pop + fronts[i]
    pop = pop[:args.pop_size]

    return pop
