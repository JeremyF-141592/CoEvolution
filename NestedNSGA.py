# NSGA2 Inspired co-evolution of policies and environments
# Author : FERSULA Jeremy

from Parameters import Configuration
from Utils.Loader import resume_from_folder, prepare_folder
from Baseline.NSGAII_core import *
from NNSGA.NNSGA_core import *
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

parser = argparse.ArgumentParser(description='NSGA2 Inspired co-evolution of policies and environments')

# General
parser.add_argument('--T', type=int, default=400, help='Iterations limit')
parser.add_argument('--resume_from', type=str, default="", help="Resume execution from folder")
parser.add_argument('--save_to', type=str, default="./NNSGA_execution", help="Execution save-to folder")
parser.add_argument('--verbose', type=int, default=0, help="Print information")
# Population
parser.add_argument('--e_init', type=str, default="flat", help='Initial policy of environments among ["flat"]')
parser.add_argument('--theta_init', type=str, default="random", help='Initial policy of individuals among ["random"]')
# NSGA2
parser.add_argument('--p_mut_ag', type=float, default=0.2, help='Probability of agent mutation')
parser.add_argument('--p_cross_ag', type=float, default=0.3, help='Probability of agent crossover')

parser.add_argument('--mut_step', type=float, default=0.1, help='Step for agent mutation')
parser.add_argument('--p_mut_gene', type=float, default=0.1, help='Probability of agent gene mutation')

parser.add_argument('--knn', type=int, default=5, help='KNN agent novelty')
parser.add_argument('--knn_env', type=int, default=5, help='KNN environment novelty')

# NNSGA
parser.add_argument('--pop_size', type=int, default=2, help='Population size on each environment')
parser.add_argument('--gen_size', type=int, default=4, help='Amount of newly generated individuals')
parser.add_argument('--pop_env_size', type=int, default=6, help='Amount of actives environments')
parser.add_argument('--pop_general_size', type=int, default=6, help='Population size on each environment')
parser.add_argument('--t_local', type=int, default=2, help='Iterations spent locally')
parser.add_argument('--t_global', type=int, default=1, help='Iterations spent globally')
parser.add_argument('--p_mean', type=float, default=-2, help='Generalisation p-mean novelty')

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
    pop_agent = list()
    prepare_folder(args)  # checks if folder exist and propose to erase it
    with open(f"{args.save_to}/commandline_args.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)


# NSGAII Algorithm -----------------------------------------------------------------------------------------------------

if ea_load:
    pop_ag = ea_load[0]
    pop_env = ea_load[1]
    pop_generalist = ea_load[2]
else:
    pop_ag = [new_population([], args) for i in range(args.pop_env_size)]
    pop_env = generate_environments([], args)
    pop_generalist = list()

front_objs = [list() for i in range(args.pop_env_size)]
for t in range(start_from, args.T):
    local = t % (args.t_local + args.t_global) < args.t_local
    gen_env = t % (args.t_local + args.t_global) == args.t_local + args.t_global - 1

    if local:
        print(f"Local iteration {t} ...")
        for i in range(args.pop_env_size):
            pop_ag[i], front_objs[i] = NSGAII(pop_ag[i], [pop_env[i]], [obj_mean_fitness, obj_genotypic_novelty], args)
    else:
        print(f"Global iteration {t} ...")
        if len(pop_generalist) == 0:
            # For each environment, extract pop_general_size / pop_env_size individuals
            for i in range(args.pop_env_size):
                c_dists = crowding_distance(front_objs[i])
                extraction_size = int(np.floor(args.pop_general_size / args.pop_env_size))
                c_sorted = np.array(c_dists).argsort()
                for k in range(extraction_size):
                    pop_generalist.append(pop_ag[i][c_sorted[k]])
        pop_generalist, _ = NSGAII(pop_generalist, pop_env, [obj_generalisation, obj_generalist_novelty], args)

    if gen_env:
        print(f"Generating new environments ...")
        proposed_environments = generate_environments(pop_env, args)
        pop_env = NSGAII_env(pop_generalist, proposed_environments, [obj_env_pata_ec, obj_env_forwarding], args)
        for i in range(args.pop_env_size):
            pop_ag[i] += pop_generalist

        pop_generalist = list()  # reset generalists

    # Save execution ----------------------------------------------------------------------------------
    with open(f'{args.save_to}/Iteration_{t}.pickle', 'wb') as f:
        pickle.dump((pop_ag, pop_env, pop_generalist), f)
    with open(f"{args.save_to}/TotalBudget.json", 'w') as f:
        budget_dic = dict()
        budget_dic["Budget_per_step"] = Configuration.budget_spent
        budget_dic["Total"] = sum(Configuration.budget_spent)
        json.dump(budget_dic, f)
    if args.verbose > 0:
        print(f"\tExecution saved at {args.save_to}.")
