# POET Enhanced Implementation as in Wang, rui and Lehman, Joel, and Clune,
# Jeff, and Stanley, Kenneth O. 2020 Uber AI Labs.
#
# Author : FERSULA Jeremy

from Parameters import Configuration
from Algorithms.POET.Mutation import mutate_envs
from Algorithms.POET.LocalTraining import ES_Step, NSGAII_step
from Algorithms.POET.Transfer import Evaluate_Candidates
from Utils.Loader import resume_from_folder, prepare_folder
from Utils.Stats import bundle_stats, append_stats
import numpy as np
import ipyparallel as ipp
import argparse
import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

Configuration.make()  # Differed imports

# Ipyparallel --------------------------------------------------------------------------------------------------
# Local parallelism, make sure that ipcluster is started beforehand otherwise this will raise an error.
Configuration.rc = ipp.Client()
with Configuration.rc[:].sync_imports():
    from Parameters import Configuration
Configuration.rc[:].execute("Configuration.make()")
Configuration.lview = Configuration.rc.load_balanced_view()
Configuration.lview.block = True

# Parse arguments ------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='POET Enhanced Implementation as in Wang, rui and Lehman, Joel, and Clune,'
                                             'Jeff, and Stanley, Kenneth O. 2020 Uber AI Labs.')

# General
parser.add_argument('--T', type=int, default=400, help='Iterations limit')
parser.add_argument('--resume_from', type=str, default="", help="Resume execution from folder.")
parser.add_argument('--save_to', type=str, default="./POET_execution", help="Execution save-to folder.")
parser.add_argument('--verbose', type=int, default=0, help="Print information.")
parser.add_argument('--max_budget', type=int, default=-1, help="Maximum number of environment evaluations.")
parser.add_argument('--save_mode', type=str, default="all", help="'all' or 'last'")
# Population
parser.add_argument('--pop_size', type=int, default=1, help='Initial population size')
# Local optimization
parser.add_argument('--lr_init', type=float, default=0.01, help="Learning rate initial value")
parser.add_argument('--lr_decay', type=float, default=0.9999, help="Learning rate decay")
parser.add_argument('--lr_limit', type=float, default=0.001, help="Learning rate limit")
parser.add_argument('--noise_std', type=float, default=0.1,  help='Noise std for local ES-optimization')
parser.add_argument('--noise_decay', type=float, default=0.999)
parser.add_argument('--noise_limit', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for ES gradient descent')
parser.add_argument('--w_decay', type=float, default=0.01, help='Weight decay penalty')
# POET
parser.add_argument('--n_mutate', type=int, default=10, help='Number of steps before attempting mutation')
parser.add_argument('--n_transfer', type=int, default=10, help='Number of steps before attempting transfer')
parser.add_argument('--max_children', type=int, default=100, help='Maximum number of children per reproduction')
parser.add_argument('--max_admitted', type=int, default=7, help='Maximum number of children admitted per reproduction')
parser.add_argument('--capacity', type=int, default=8, help='Maximum number of active environments')
parser.add_argument('--repro_threshold', type=int, default=200, help='Minimum score to be allowed to reproduce')

parser.add_argument('--mc_min', type=int, default=-25, help='Minimal fitness to pass MC')
parser.add_argument('--mc_max', type=int, default=340, help='Maximal fitness to pass MC')

parser.add_argument('--knn', type=int, default=5, help='Amount of neighbors evaluating knn Environment Novelty')


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

if folder != "":
    ea_list_resume, start_from = resume_from_folder(folder, args)
else:
    prepare_folder(args)  # checks if folder exist and propose to erase it

# POET Algorithm -------------------------------------------------------------------------------------------------------
# This part is intended to be as close as possible as the pseudo-code presented in the original paper.

threshold = np.zeros((args.capacity, 5))  # List of fitness thresholds, needed for transfer

EA_List = list()
if folder == "":
    # Generate new pairs agent, environment
    EA_List = [(Configuration.envFactory.new(), Configuration.agentFactory.new()) for i in range(args.pop_size)]
else:
    EA_List = ea_list_resume
for t in range(start_from, args.T):
    print(f"Iteration {t} ...", end=" ", flush=True)
    Configuration.budget_spent.append(0)

    # Mutation --------------------------------------------------------
    if t > 0 and t % args.n_mutate == 0:
        print("Mutate ...", end=" ", flush=True)
        EA_List = mutate_envs(EA_List, args)

    # Local optimization ----------------------------------------------
    M = len(EA_List)
    for m in range(M):
        E, theta = EA_List[m]
        if args.verbose > 0:
            print(f"\n\t{m} : ", end="", flush=True)
        theta, threshold[m, t % 5] = ES_Step(theta, E, args, allow_verbose=1)
        # theta, threshold[m, t % 5] = ES_Step(theta, E, args, allow_verbose=1)
        EA_List[m] = (E, theta)

    # Transfer  -------------------------------------------------------
    if M > 1 and t > 0 and t % args.n_transfer == 0 and t % args.n_mutate != 0:
        if args.verbose > 0:
            print("Transfer ...")
        else:
            print("Transfer ...", end=" ", flush=True)
        new_ea_list = []
        for m in range(M):
            E, theta = EA_List[m]
            thres = threshold[m].max()  # Max fitness over 5 last runs
            theta_top, score_top = Evaluate_Candidates(EA_List[:m] + EA_List[m+1:], E, args, threshold=thres)
            if score_top > thres:
                if args.verbose > 0:
                    print(f"Transfered onto {m} : {score_top} > {thres}.")
                theta_top.set_opt_state(Configuration.optimizer.default_state())
                new_ea_list.append((E, theta_top))
            else:
                new_ea_list.append((E, theta))
        EA_List = new_ea_list

    print(" Done.")

    # Save current execution ------------------------------------------
    if args.save_mode == "last" and t > 0:
        os.remove(f'{args.save_to}/Iteration_{t-1}.pickle')
    with open(f'{args.save_to}/Iteration_{t}.pickle', 'wb') as f:
        pickle.dump(EA_List, f)
    with open(f'{args.save_to}/Archive.pickle', 'wb') as f:
        pickle.dump(Configuration.archive, f)
    with open(f"{args.save_to}/TotalBudget.json", 'w') as f:
        budget_dic = dict()
        budget_dic["Budget_per_step"] = Configuration.budget_spent
        budget_dic["Total"] = sum(Configuration.budget_spent)
        json.dump(budget_dic, f)
    bundle = bundle_stats([i[1] for i in EA_List], [i[0] for i in EA_List])
    append_stats(f"{args.save_to}/Stats.json", bundle)
    if args.verbose > 0:
        print(f"\tExecution saved at {args.save_to}.")
    if 0 < args.max_budget < sum(Configuration.budget_spent):
        print(f"\nMaximum budget exceeded : {sum(Configuration.budget_spent)} > {args.max_budget}.\n")
        break
