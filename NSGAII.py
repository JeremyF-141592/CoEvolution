from Parameters import Configuration
from Utils.Loader import resume_from_folder, prepare_folder
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
parser.add_argument('--nb_rounds', type=int, default=1, help='Number of episodes to evaluate any agent')
# Population
parser.add_argument('--e_init', type=str, default="flat", help='Initial policy of environments among ["flat"]')
parser.add_argument('--theta_init', type=str, default="random", help='Initial policy of individuals among ["random"]')
parser.add_argument('--pop_size', type=int, default=100, help='Population size')
# NSGA2
parser.add_argument('--env_path', type=str, default="./Baseline/NSGA_env.pickle", help='Path to the pickled environment')
parser.add_argument('--p_mut_ag', type=float, default=0.2, help='Probability of mutation')
parser.add_argument('--p_cross_ag', type=float, default=0.3, help='Probability of crossover')

parser.add_argument('--mut_step', type=float, default=0.1, help='Step for agent mutation')
parser.add_argument('--p_mut_gene', type=float, default=0.1, help='Probability of agent gene mutation')

args = parser.parse_args()

# Resume execution -----------------------------------------------------------------------------------------------------

folder = ""
start_from = 0
ea_list_resume = []
if args.resume_from != "":
    #  if we load arguments, args is going to change so we need a variable to store the folder name
    folder = args.resume_from

if folder != "":
    ea_list_resume, start_from = resume_from_folder(folder, args)
else:
    prepare_folder(args)  # checks if folder exist and propose to erase it
    with open(f"{args.save_to}/commandline_args.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

Configuration.nb_rounds = args.nb_rounds

# NSGAII Algorithm -----------------------------------------------------------------------------------------------------

with open(args.env_path, "rb") as f:
    env = pickle.load(f)

pop = list()
for t in range(start_from, args.T):
    if args.verbose > 0:
        print(f"Iteration {t} ...", flush=True)

    new_pop = new_population(pop, args)
    pop = list()

    results = Configuration.lview.map(env, new_pop)

    nd_sort = np.array(fast_non_dominated_sort(results))

    fronts = [list() for i in range(nd_sort.max() + 1)]

    for i in range(len(new_pop)):
        fronts[nd_sort[i]].append(new_pop[i])

    count = -1
    for i in range(len(fronts)):
        if len(pop) > args.pop_size:
            break
        pop = pop + fronts[i]
        count += 1
    pop = pop[:args.pop_size]

    if args.verbose > 0:
        print(f"\tOne of the first agent has objectives : {results[nd_sort.argmin()]}", flush=True)

    # Save execution ----------------------------------------------------------------------------------
    with open(f'{args.save_to}/Iteration_{t}.pickle', 'wb') as f:
        pickle.dump(pop, f)
    with open(f"{args.save_to}/TotalBudget.json", 'w') as f:
        budget_dic = dict()
        budget_dic["Budget_per_step"] = Configuration.budget_spent
        budget_dic["Total"] = sum(Configuration.budget_spent)
        json.dump(budget_dic, f)
    if args.verbose > 0:
        print(f"\tExecution saved at {args.save_to}.")
