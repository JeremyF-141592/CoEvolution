# NSGA2 Inspired co-evolution of policies and environments
# Author : FERSULA Jeremy

from Utils.Loader import resume_from_folder, prepare_folder
from Utils.Stats import append_stats
from Algorithms.NSGA2.NSGAII_tools import *
import numpy as np
import ipyparallel as ipp
import argparse
import json
import os
import pickle
import warnings
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
parser.add_argument('--save_to', type=str, default="./CoEvoNSGA_execution", help="Execution save-to folder")
parser.add_argument('--verbose', type=int, default=0, help="Print information")
parser.add_argument('--max_budget', type=int, default=-1, help="Maximum number of environment evaluations.")
parser.add_argument('--save_mode', type=str, default="all", help="'all' or 'last'")
# Population
parser.add_argument('--e_init', type=str, default="flat", help='Initial policy of environments among ["flat"]')
parser.add_argument('--theta_init', type=str, default="random", help='Initial policy of individuals among ["random"]')
# NSGA2
parser.add_argument('--p_mut_ag', type=float, default=0.2, help='Probability of agent mutation')
parser.add_argument('--p_cross_ag', type=float, default=0.3, help='Probability of agent crossover')

parser.add_argument('--p_mut_gene', type=float, default=0.1, help='Probability of agent gene mutation')
parser.add_argument('--eta_mut', type=float, default=0.5, help='Eta in polynomial bounded mutation')
parser.add_argument('--eta_cross', type=float, default=0.5, help='Eta in SimulatedBinary crossover')

parser.add_argument('--knn', type=int, default=5, help='KNN agent novelty')
parser.add_argument('--knn_env', type=int, default=5, help='KNN environment novelty')

# NNSGA
parser.add_argument('--pop_size', type=int, default=100, help='Population size on each environment')
parser.add_argument('--gen_size', type=int, default=100, help='Amount of newly generated individuals')
parser.add_argument('--quantile', type=float, default=0.3, help='Generalisation score, quantile of fitness')
parser.add_argument('--p_mut_env', type=float, default=0.25, help='Probability of environment mutation')

parser.add_argument('--max_env_children', type=int, default=100, help='Maximum number of env children per reproduction')

parser.add_argument('--pata_ec_tol', type=float, default=2, help='Ranking tolerance for PATA_EC diversity')
parser.add_argument('--pata_ec_clipmax', type=float, default=250, help='Upper fitness bound for PATA_EC diversity')
parser.add_argument('--pata_ec_clipmin', type=float, default=-50, help='Lower fitness bound for PATA_EC diversity')

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

# ----------------------------------------------------------------------------------------------------------------------

def evaluate_pair(ea):
    e, a = ea
    return e(a)

def NSGAII_ag_env(pop, envs, args):

    new_pop = pop + new_population(pop, args)
    new_evs = envs + generate_env(envs, args)

    for p in new_pop:
        if type(p.get_opt_state) != int:
            p.set_opt_state(0)
        else:
            p.set_opt_state(p.get_opt_state() + 1)

    pairs = [(ev, ag) for ev, ag in zip(new_evs, new_pop)]

    if len(new_evs) == 1:
        new_evs = generate_env(new_evs, args)

    res = Configuration.lview.map(evaluate_pair, pairs)
    Configuration.budget_spent[-1] += len(res)
    if type(res[0]) != tuple and type(res[0]) != list:
        raise TypeError("Current fitness metric returns a scalar instead of a tuple.")

    results = [[res[i][0] - new_pop[i].get_opt_state(), res[i][1]] for i in range(len(res))]

    nd_sort = np.array(fast_non_dominated_sort(results))

    fronts = [list() for i in range(nd_sort.max() + 1)]  # store agents
    fronts_evs = [list() for i in range(nd_sort.max() + 1)]  # store environments
    fronts_objectives = [list() for i in range(nd_sort.max() + 1)]  # store agents objectives
    for i in range(len(new_pop)):
        fronts[nd_sort[i]].append(new_pop[i])
        fronts_evs[nd_sort[i]].append(new_evs[i])
        fronts_objectives[nd_sort[i]].append(results[i])

    pop = list()  # New population
    evs = list()   # New env population
    objs = list()  # Corresponding objectives
    last_front = 0
    for i in range(len(fronts)):
        if len(pop) + len(fronts[i]) > args.pop_size:
            break
        pop = pop + fronts[i]
        evs = evs + fronts_evs[i]
        objs = objs + fronts_objectives[i]
        last_front = i + 1

    if last_front < len(fronts):
        cdistance = np.array(crowding_distance(fronts_objectives[last_front]))
        cdist_sort = cdistance.argsort()[::-1]

        # print("Keep", last_front, "fronts and add", args.pop_size - len(pop), "individuals via crowding distance.")

        for i in range(args.pop_size - len(pop)):  # fill the population with less crowded individuals of the last front
            pop.append(fronts[last_front][cdist_sort[i]])
            evs.append(fronts_evs[last_front][cdist_sort[i]])
            objs.append(fronts_objectives[last_front][cdist_sort[i]])

    # Return new population and their objectives
    return pop, evs, objs


def generate_env(envs, args):
    """Generate new environments by mutating old environments"""
    new_list = list()
    if len(envs) == 0:
        new_list.append(Configuration.envFactory.new())
        return new_list
    for i in range(args.max_env_children):
        choice = np.random.randint(0, len(envs))
        if np.random.uniform(0, 1) < args.p_mut_env:
            new_list.append(envs[choice].get_child())
        else:
            new = Configuration.envFactory.new()
            new.__setstate__(envs[choice].__getstate__())
            new_list.append(new)
    return new_list


def obj_paired_fitness_age(index, fitness, observation, new_pop, envs, args):
    # p-mean over environments fitness
    age = new_pop[index].get_opt_state()
    return fitness[index][index] - age


# NSGAII Algorithm -----------------------------------------------------------------------------------------------------

if ea_load:
    pop_ag = ea_load[0]
    pop_env = ea_load[1]
else:
    pop_ag = list()
    pop_env = list()

objs_local = [list() for i in range(len(pop_env))]
objs_general = list()
for t in range(start_from, args.T):
    Configuration.budget_spent.append(0)
    print(f"Iteration {t}", flush=True)

    pop_ag, pop_env, objs_local = NSGAII_ag_env(pop_ag, pop_env, args)

    # Save execution ----------------------------------------------------------------------------------
    if args.save_mode == "last" and t > 0:
        os.remove(f'{args.save_to}/Iteration_{t-1}.pickle')
    with open(f'{args.save_to}/Iteration_{t}.pickle', 'wb') as f:
        pickle.dump((pop_ag, pop_env), f)
    with open(f"{args.save_to}/TotalBudget.json", 'w') as f:
        budget_dic = dict()
        budget_dic["Budget_per_step"] = Configuration.budget_spent
        budget_dic["Total"] = sum(Configuration.budget_spent)
        json.dump(budget_dic, f)

    bundle = dict()
    for k in range(len(objs_local[0])):
        bundle[f"Objective_{k}-max"] = list()

    for k in range(len(objs_local[0])):  # reformat objectives from list of tuple to lists for each objective
        obj_list = list()
        for j in range(len(objs_local)):
            obj_list.append(objs_local[j][k])
        obj_arr = np.array(obj_list)
        bundle[f"Objective_{k}-max"].append(obj_arr.max())

    append_stats(f"{args.save_to}/Stats.json", bundle)
    if args.verbose > 0:
        print(f"\tExecution saved at {args.save_to}.")
    if 0 < args.max_budget < sum(Configuration.budget_spent):
        print(f"\nMaximum budget exceeded : {sum(Configuration.budget_spent)} > {args.max_budget}.\n")
        break
