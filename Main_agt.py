# NSGA2 Inspired co-evolution of policies and environments
# Author : FERSULA Jeremy

from Utils.Loader import resume_from_folder, prepare_folder
from Utils.Stats import append_stats, bundle_stats
from Algorithms.NSGA2.NSGAII_tools import *
import numpy as np
import ipyparallel as ipp
import argparse
import json
import os
import pickle
import warnings
from collections import deque
warnings.filterwarnings("ignore")

Configuration.make()
# Ipyparallel --------------------------------------------------------------------------------------------------
# Local parallelism, make sure that ipcluster is started beforehand otherwise this will raise an error.
Configuration.rc = ipp.Client()
with Configuration.rc[:].sync_imports():
    from Parameters import Configuration
Configuration.rc[:].execute("Configuration.make()")
Configuration.lview = Configuration.rc.load_balanced_view()
Configuration.lview.block = True

# Parse arguments ------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='NSGA2 Inspired co-evolution of policies and environments')

# General
parser.add_argument('--T', type=int, default=400, help='Iterations limit')
parser.add_argument('--resume_from', type=str, default="", help="Resume execution from folder")
parser.add_argument('--save_to', type=str, default="./agt_execution", help="Execution save-to folder")
parser.add_argument('--verbose', type=int, default=0, help="Print information")
parser.add_argument('--max_budget', type=int, default=-1, help="Maximum number of environment evaluations.")
parser.add_argument('--save_mode', type=str, default="all", help="Specify save mode among ['all', 'last', N] where N is"
                                                                 "a number corresponding the saving's interval.")
# Population
parser.add_argument('--e_init', type=str, default="flat", help='Initial policy of environments among ["flat"]')
parser.add_argument('--theta_init', type=str, default="random", help='Initial policy of individuals among ["random"]')
# NSGA2
parser.add_argument('--p_mut_ag', type=float, default=0.5, help='Probability of agent mutation')
parser.add_argument('--p_cross_ag', type=float, default=0, help='Probability of agent crossover')

parser.add_argument('--mut_low_bound', type=float, default=-1.0, help='Lower bound for polynomial bounded mutation')
parser.add_argument('--mut_high_bound', type=float, default=1.0, help='Upper bound for polynomial bounded mutation')

parser.add_argument('--p_mut_gene', type=float, default=0.1, help='Probability of agent gene mutation')
parser.add_argument('--eta_mut', type=float, default=0.5, help='Eta in polynomial bounded mutation')
parser.add_argument('--eta_cross', type=float, default=0.5, help='Eta in SimulatedBinary crossover')

parser.add_argument('--knn', type=int, default=5, help='KNN agent novelty')
parser.add_argument('--knn_env', type=int, default=5, help='KNN environment novelty')

# NNSGA
parser.add_argument('--pop_size', type=int, default=50, help='Population size on each environment')
parser.add_argument('--gen_size', type=int, default=50, help='Amount of newly generated individuals')
parser.add_argument('--pop_env_size', type=int, default=10, help='Amount of actives environments')

parser.add_argument('--progress_min', type=int, default=5, help='Minimum amount of iterations before discard an'
                                                                'environment.')

args = parser.parse_args()

# Resume execution -----------------------------------------------------------------------------------------------------

folder = ""
start_from = 0
ea_list_resume = []
if args.resume_from != "":
    #  if we load arguments, args is going to change so we need a variable to store the folder name
    folder = args.resume_from

ea_load = None
if folder != "":
    ea_load, start_from = resume_from_folder(folder, args)
else:
    prepare_folder(args)  # checks if folder exist and propose to erase it


def NSGAII_agt(pop, envs, additional_objectives, args):

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

    valid_environments = list()
    for i in range(len(envs)):
        ev = envs[i]
        if not hasattr(ev, "scores"):
            ev.scores = deque()
        ev.scores.append(np.array(fit[i]).max())
        if len(ev.scores) > args.progress_min:
            ev.scores.popleft()
            valid_environments.append(i)

    minimum = float("inf")
    argmin_progress = 0
    for i in valid_environments:
        p = envs[i].scores[-1] - envs[i].scores[0]
        if p < minimum:
            minimum = p
            argmin_progress = i
    envs.remove(envs[argmin_progress])

    new_env = envs[np.random.randint(0, len(envs))].get_child()
    envs.append(new_env)

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


# NSGAII Algorithm -----------------------------------------------------------------------------------------------------

if ea_load:
    pop_env = ea_load[0]
    pop_generalist = ea_load[1]
else:
    pop_env = [Configuration.envFactory.new().get_child() for i in range(args.pop_env_size)]
    pop_generalist = new_population([], args)


objs_general = list()
for t in range(start_from, args.T):
    Configuration.budget_spent.append(0)

    print(f"Global iteration {t} ...")
    pop_generalist, objs_general = NSGAII_agt(pop_generalist, pop_env, [obj_mean_fitness, obj_generalist_novelty], args)
    pop_generalist = [pop_generalist[i] for _, i in sorted(zip(objs_general, range(len(objs_general))))]

    # Save execution ----------------------------------------------------------------------------------
    remove_previous = False
    if args.save_mode == "last" and t > 0:
        remove_previous = True
    if args.save_mode.isdigit():
        remove_previous = True
        if t % int(args.save_mode) == 0:
            remove_previous = False
    if remove_previous:
        os.remove(f'{args.save_to}/Iteration_{t-1}.pickle')
    with open(f'{args.save_to}/Iteration_{t}.pickle', 'wb') as f:
        pickle.dump((pop_env, pop_generalist), f)
    with open(f"{args.save_to}/TotalBudget.json", 'w') as f:
        budget_dic = dict()
        budget_dic["Budget_per_step"] = Configuration.budget_spent
        budget_dic["Total"] = sum(Configuration.budget_spent)
        json.dump(budget_dic, f)

    bundle = bundle_stats(pop_generalist, pop_env)
    additional_stats = bundle_stats_NNSGA(False, [], objs_general, args)

    bundle.update(additional_stats)

    append_stats(f"{args.save_to}/Stats.json", bundle)
    if args.verbose > 0:
        print(f"\tExecution saved at {args.save_to}.")
    if 0 < args.max_budget < sum(Configuration.budget_spent):
        print(f"\nMaximum budget exceeded : {sum(Configuration.budget_spent)} > {args.max_budget}.\n")
        break
