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
parser.add_argument('--save_to', type=str, default="./NNSGA_execution", help="Execution save-to folder")
parser.add_argument('--verbose', type=int, default=0, help="Print information")
parser.add_argument('--max_budget', type=int, default=-1, help="Maximum number of environment evaluations.")
parser.add_argument('--save_mode', type=str, default="all", help="'all' or 'last'")
# Population
parser.add_argument('--e_init', type=str, default="flat", help='Initial policy of environments among ["flat"]')
parser.add_argument('--theta_init', type=str, default="random", help='Initial policy of individuals among ["random"]')
# NSGA2
parser.add_argument('--p_mut_ag', type=float, default=0.2, help='Probability of agent mutation')
parser.add_argument('--p_cross_ag', type=float, default=0.3, help='Probability of agent crossover')

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
parser.add_argument('--pop_env_size', type=int, default=6, help='Amount of actives environments')
parser.add_argument('--pop_general_size', type=int, default=50, help='Population size on each environment')
parser.add_argument('--t_local', type=int, default=20, help='Iterations spent locally')
parser.add_argument('--t_global', type=int, default=20, help='Iterations spent globally')
parser.add_argument('--mean', type=float, default=-0.25, help='Generalisation score, sliding mean of fitness')

parser.add_argument('--max_env_children', type=int, default=100, help='Maximum number of env children per reproduction')
parser.add_argument('--load_env', type=str, default="", help='Whether to load a predefined set of environments.')

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


# NSGAII Algorithm -----------------------------------------------------------------------------------------------------

if ea_load:
    pop_ag = ea_load[0]
    pop_env = ea_load[1]
    pop_generalist = ea_load[2]
else:
    pop_env = generate_environments([], args)
    if args.load_env != "":
        with open(args.load_env, "rb") as f:
            pop_env = pickle.load(f)
        args.pop_env_size = len(pop_env)
    pop_ag = [new_population([], args) for i in range(args.pop_env_size)]
    pop_generalist = new_population([], args)

objs_local = [list() for i in range(len(pop_env))]
objs_general = list()
for t in range(start_from, args.T):
    Configuration.budget_spent.append(0)
    local = t % (args.t_local + args.t_global) < args.t_local
    gen_env = t % (args.t_local + args.t_global) == 0 and t != 0 and args.load_env == ""
    transition_global = t % (args.t_local + args.t_global) == args.t_local and args.t_local != 0

    if gen_env:
        print(f"Generating new environments ...")
        proposed_environments = generate_environments(pop_env, args)
        pop_env = NSGAII_env(pop_generalist, proposed_environments, [obj_parametrized_env_novelty, obj_env_forwarding], args)
        for i in range(len(pop_env)):
            pop_ag[i] += pop_generalist
        objs_local = [list() for i in range(len(pop_env))]

    if local:
        print(f"Local iteration {t} ...")
        for i in range(len(pop_env)):
            pop_ag[i], objs_local[i] = NSGAII(pop_ag[i], [pop_env[i]], [obj_mean_fitness, obj_mean_observation_novelty],args)
            pop_ag[i] = [pop_ag[i][j] for _, j in sorted(zip(objs_local[i], range(len(objs_local[i]))))]
    else:
        print(f"Global iteration {t} ...")
        if transition_global:
            # For each environment, extract pop_general_size / pop_env_size individuals
            for i in range(len(pop_env)):
                c_dists = crowding_distance(objs_local[i])
                extraction_size = int(np.floor(args.pop_general_size / len(pop_env)))
                c_sorted = np.array(c_dists).argsort()
                for k in range(min(extraction_size, args.pop_size)):
                    pop_generalist.append(pop_ag[i][c_sorted[k]])
        pop_generalist, objs_general = NSGAII(pop_generalist, pop_env, [obj_generalisation, obj_generalist_novelty], args)
        pop_generalist = [pop_generalist[i] for _, i in sorted(zip(objs_general, range(len(objs_general))))]

    # Save execution ----------------------------------------------------------------------------------
    if args.save_mode.isdigit():
        if t % int(args.save_mode) == 0:
            with open(f'{args.save_to}/Iteration_{t}.pickle', 'wb') as f:
                pickle.dump((pop_ag, pop_env, pop_generalist), f)
    else:
        if args.save_mode == "last" and t > 0:
            os.remove(f'{args.save_to}/Iteration_{t-1}.pickle')
        with open(f'{args.save_to}/Iteration_{t}.pickle', 'wb') as f:
            pickle.dump((pop_ag, pop_env, pop_generalist), f)
    with open(f"{args.save_to}/TotalBudget.json", 'w') as f:
        budget_dic = dict()
        budget_dic["Budget_per_step"] = Configuration.budget_spent
        budget_dic["Total"] = sum(Configuration.budget_spent)
        json.dump(budget_dic, f)

    bundle = bundle_stats_NNSGA(local, objs_local, objs_general, args)
    # Benchmark saving
    if Configuration.benchmark is not None:
        bundle["xy_benchmark"] = list()
        if local:
            for i in range(len(pop_env)):
                for j in range(args.pop_size):
                    bundle["xy_benchmark"].append((pop_ag[i][j].value, pop_env[i].y_value))
        else:
            for i in range(len(pop_env)):
                for j in range(len(pop_generalist)):
                    bundle["xy_benchmark"].append((pop_generalist[j].value, pop_env[i].y_value))

    append_stats(f"{args.save_to}/Stats.json", bundle)
    if args.verbose > 0:
        print(f"\tExecution saved at {args.save_to}.")
    if 0 < args.max_budget < sum(Configuration.budget_spent):
        print(f"\nMaximum budget exceeded : {sum(Configuration.budget_spent)} > {args.max_budget}.\n")
        break
