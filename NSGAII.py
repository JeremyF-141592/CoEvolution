# NSGA2 Implementation as in Deb, K., Pratap, A., Agarwal, S., & Meyarivan,
# T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II
#
# Author : FERSULA Jeremy

from Parameters import Configuration
from Utils.Loader import resume_from_folder, prepare_folder
from Utils.Stats import bundle_stats, append_stats
from Baseline.NSGAII_core import *
import numpy as np
import ipyparallel as ipp
import argparse
import json
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

parser = argparse.ArgumentParser(description='NSGA2 Implementation as in Deb, K., Pratap, A., Agarwal, S., & Meyarivan,'
                                             'T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: '
                                             'NSGA-II')

# General
parser.add_argument('--T', type=int, default=400, help='Iterations limit')
parser.add_argument('--resume_from', type=str, default="", help="Resume execution from folder.")
parser.add_argument('--save_to', type=str, default="./NSGA_execution", help="Execution save-to folder.")
parser.add_argument('--verbose', type=int, default=0, help="Print information.")
parser.add_argument('--max_budget', type=int, default=-1, help="Maximum number of environment evaluations.")
# Population
parser.add_argument('--pop_size', type=int, default=25, help='Population size')
# NSGA2
parser.add_argument('--gen_size', type=int, default=75, help='Population generation size')
parser.add_argument('--env_path', type=str, default="./Baseline/NSGA_env.pickle", help='Path to the pickled environment')
parser.add_argument('--p_mut_ag', type=float, default=0.2, help='Probability of mutation')
parser.add_argument('--p_cross_ag', type=float, default=0.3, help='Probability of crossover')

parser.add_argument('--eta_mut', type=float, default=0.5, help='Eta in polynomial bounded mutation')
parser.add_argument('--eta_cross', type=float, default=0.5, help='Eta in SimulatedBinary crossover')
parser.add_argument('--p_mut_gene', type=float, default=0.1, help='Probability of agent gene mutation')

parser.add_argument('--knn', type=int, default=5, help='KNN novelty')

args = parser.parse_args()

# Resume execution -----------------------------------------------------------------------------------------------------

folder = ""
start_from = 0
pop = list()
if args.resume_from != "":
    #  if we load arguments, args is going to change so we need a variable to store the folder name
    folder = args.resume_from

if folder != "":
    pop, start_from = resume_from_folder(folder, args)
else:
    prepare_folder(args)  # checks if folder exist and propose to erase it
    with open(f"{args.save_to}/commandline_args.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)


# NSGAII Algorithm -----------------------------------------------------------------------------------------------------

with open(args.env_path, "rb") as f:
    env = pickle.load(f)

for t in range(start_from, args.T):
    print(f"Iteration {t} ...", flush=True)
    Configuration.budget_spent.append(0)

    new_pop = pop + new_population(pop, args)

    results = Configuration.lview.map(env, new_pop)

    Configuration.budget_spent[-1] += len(results)

    # GENOTYPIC NOVELTY ---- todo : clean up, add the option of BC novelty
    for i in range(len(results)):
        w = new_pop[i].get_weights()
        n_score = 0
        dists = np.zeros(len(new_pop))
        for j in range(len(new_pop)):
            dists[j] = np.linalg.norm(w - new_pop[j].get_weights())
        dists.sort()
        results[i] = (results[i][0], dists[:args.knn].mean())

    nd_sort = np.array(fast_non_dominated_sort(results))

    fronts = [list() for i in range(nd_sort.max() + 1)]             # store agents
    fronts_objectives = [list() for i in range(nd_sort.max() + 1)]  # store agents objectives

    for i in range(len(new_pop)):
        fronts[nd_sort[i]].append(new_pop[i])
        fronts_objectives[nd_sort[i]].append(results[i])

    pop = list()    # New population
    objs = list()   # Corresponding objectives
    last_front = 0
    for i in range(len(fronts)):
        if len(pop) + len(fronts[i]) > args.pop_size:
            break
        pop = pop + fronts[i]
        objs = objs + fronts_objectives[i]
        last_front = i + 1

    cdistance = np.array(crowding_distance(fronts_objectives[last_front]))
    cdist_sort = cdistance.argsort()[::-1]

    print("Keep", last_front, "fronts and add", args.pop_size - len(pop), "individuals via crowding distance.")

    for i in range(args.pop_size - len(pop)):  # fill the population with less crowded individuals of the last front
        pop.append(fronts[last_front][cdist_sort[i]])
        objs.append(fronts_objectives[last_front][cdist_sort[i]])

    # Save execution ----------------------------------------------------------------------------------
    with open(f'{args.save_to}/Iteration_{t}.pickle', 'wb') as f:
        pickle.dump(pop, f)
    with open(f"{args.save_to}/TotalBudget.json", 'w') as f:
        budget_dic = dict()
        budget_dic["Budget_per_step"] = Configuration.budget_spent
        budget_dic["Total"] = sum(Configuration.budget_spent)
        json.dump(budget_dic, f)
    bundle = bundle_stats(pop, [env])
    for k in range(len(objs[0])):   # reformat objectives from list of tuple to lists for each objective
        bundle[f"Objective_{k}"] = list()
        for i in range(len(objs)):
            bundle[f"Objective_{k}"].append(objs[i][k])
    append_stats(f"{args.save_to}/Stats.json", bundle)
    if args.verbose > 0:
        print(f"\tExecution saved at {args.save_to}.")
    if 0 < args.max_budget < sum(Configuration.budget_spent):
        print(f"\nMaximum budget exceeded : {sum(Configuration.budget_spent)} > {args.max_budget}.\n")
        break
