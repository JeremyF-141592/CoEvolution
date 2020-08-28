"""
NSGA2 (Deb et al. 2002), evaluated by default on a random set of 20 environments at each iteration.
The environment set can be specified with a pickle file, using --load_env.
"""
import argparse
import json
import pickle
import warnings
import os
import ipyparallel as ipp
from Utils.Loader import resume_from_folder, prepare_folder
from Utils.Stats import bundle_stats, append_stats
from Algorithms.NSGA2.NSGAII_tools import *
from Parameters import Configuration
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
parser.add_argument('--save_to', type=str, default="./NSGA2_execution", help="Execution save-to folder.")
parser.add_argument('--save_mode', type=str, default="all", help="Specify save mode among ['all', 'last', N] where N is"
                                                                 "a number corresponding the saving's interval.")
parser.add_argument('--verbose', type=int, default=0, help="Print information.")
parser.add_argument('--max_budget', type=int, default=-1, help="Maximum number of environment evaluations.")
# Population
parser.add_argument('--pop_size', type=int, default=100, help='Population size')
# NSGA2
parser.add_argument('--load_env', type=str, default="", help='Path to pickled environment')
parser.add_argument('--gen_size', type=int, default=100, help='Population generation size')
parser.add_argument('--p_mut_ag', type=float, default=0.5, help='Probability of mutation')
parser.add_argument('--p_cross_ag', type=float, default=0, help='Probability of crossover')

parser.add_argument('--mut_low_bound', type=float, default=-1.0, help='Lower bound for polynomial bounded mutation')
parser.add_argument('--mut_high_bound', type=float, default=1.0, help='Upper bound for polynomial bounded mutation')

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


# NSGAII Algorithm -----------------------------------------------------------------------------------------------------
envs = list()

default = True
if os.path.exists(args.load_env):
    with open(args.load_env, "rb") as f:
        envs = pickle.load(f)
    default = False

for t in range(start_from, args.T):
    print(f"Iteration {t} ...", flush=True)
    Configuration.budget_spent.append(0)

    if default:
        envs = list()
        for i in range(20):
            ev = Configuration.envFactory.new()
            for j in range(30):
                ev = ev.get_child()
            envs.append(ev)

    pop, objs = NSGAII(pop, envs, [obj_mean_fitness, obj_mean_observation_novelty], args)

    # Save execution ----------------------------------------------------------------------------------
    remove_previous = False
    if args.save_mode == "last" and t > 0:
        remove_previous = True
    if args.save_mode.isdigit():
        remove_previous = True
        if t % int(args.save_mode) == 0:
            remove_previous = False
    if remove_previous:
        os.remove(f'{args.save_to}/Iteration_{t - 1}.pickle')
    with open(f'{args.save_to}/Iteration_{t}.pickle', 'wb') as f:
        pickle.dump(pop, f)
    with open(f"{args.save_to}/TotalBudget.json", 'w') as f:
        budget_dic = dict()
        budget_dic["Budget_per_step"] = Configuration.budget_spent
        budget_dic["Total"] = sum(Configuration.budget_spent)
        json.dump(budget_dic, f)
    bundle = bundle_stats(pop, envs)
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
