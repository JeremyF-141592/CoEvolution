from IPCA.IPCA_core import *
from Templates.Agents import AgentFactory, Agent
from Utils.Loader import resume_from_folder, prepare_folder
from Utils.Stats import bundle_stats, append_stats
import ipyparallel as ipp
import pickle
import argparse
import json
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

# Check Parameters.py --------------------------------------------------------------------------------------------------

if not isinstance(Configuration.agentFactory, AgentFactory):
    raise RuntimeError("Configuration agentFactory is not an instance of AgentFactory.")
if not isinstance(Configuration.agentFactory.new(), Agent):
    raise RuntimeError("Configuration agentFactory.new() is not an instance of Agent.")
# if not issubclass(Configuration.baseEnv, EnvironmentInterface):
#     raise RuntimeError("Configuration baseEnv is not inherited from EnvironmentInterface.")

# Parse arguments ------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='nothing yet')

# General
parser.add_argument('--T', type=int, default=400, help='Iterations limit')
parser.add_argument('--resume_from', type=str, default="", help="Resume execution from folder.")
parser.add_argument('--save_to', type=str, default="./IPCA_execution", help="Execution save-to folder.")
parser.add_argument('--verbose', type=int, default=0, help="Print information.")
# Population
parser.add_argument('--e_init', type=str, default="flat", help='Initial policy of environments among ["flat"]')
parser.add_argument('--theta_init', type=str, default="random", help='Initial policy of individuals among ["random"]')
# IPCA
parser.add_argument('--nb_best', type=int, default=4, help='Nb of best agents kept in archive')
parser.add_argument('--new_tests', type=int, default=4, help='Nb of new generated tests each iteration')
parser.add_argument('--new_ags', type=int, default=16, help='Nb of new generated agents each iteration')
parser.add_argument('--p_mut_env', type=float, default=0.2, help='Probability of mutation')
parser.add_argument('--p_cross_env', type=float, default=0.3, help='Probability of crossover')
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


Learners = list()
Tests = list()

L_scores = list()
T_scores = list()

for t in range(start_from, args.T):
    if args.verbose > 0:
        print(f"Iteration {t} ...", flush=True)

    Learners = non_dominated(Learners, L_scores)
    Learners_gen = generate_learners(Learners, args)

    Tests_gen = generate_tests(Tests, args)

    prev_tests = len(Tests)
    prev_learners = len(Learners)

    L_scores, T_scores, maximum = cross_evaluation(Learners + Learners_gen, Tests + Tests_gen, args)

    if args.verbose > 0:
        print("\tOverall maximum fitness :", maximum)

    Tests_kept, indices_t = xuseful(Tests_gen, T_scores[prev_tests:], T_scores[:prev_tests])

    Learners_kept, indices = useful(Learners_gen, L_scores[prev_learners:], L_scores[:prev_learners])

    Learners += Learners_kept
    newL_scores = L_scores[:prev_learners]
    for i in indices:
        newL_scores.append(L_scores[prev_learners + i])
    L_scores = newL_scores

    Tests += Tests_kept
    newT_scores = T_scores[:prev_tests]
    for i in indices_t:
        newT_scores.append(T_scores[prev_tests + i])
    T_scores = newT_scores

    if args.verbose > 0:
        print(f"\tCurrently {len(Learners)} learners in {len(Tests)} tests.", flush=True)
    # if args.verbose > 0:
    #     print(f"Trimmed to {len(Learners)} learners in {len(Tests)} tests.", flush=True)

    # Save execution ----------------------------------------------------------------------------------
    with open(f'{args.save_to}/Iteration_{t}.pickle', 'wb') as f:
        pickle.dump((Learners, Tests), f)
    with open(f'{args.save_to}/Archive.pickle', 'wb') as f:
        pickle.dump(Configuration.archive, f)
    with open(f"{args.save_to}/TotalBudget.json", 'w') as f:
        budget_dic = dict()
        budget_dic["Budget_per_step"] = Configuration.budget_spent
        budget_dic["Total"] = sum(Configuration.budget_spent)
        json.dump(budget_dic, f)
    bundle = bundle_stats(Learners, Tests)
    bundle["Max_fit"] = maximum
    append_stats(f"{args.save_to}/Stats.json", bundle)
    if args.verbose > 0:
        print(f"\tExecution saved at {args.save_to}.")


"""
Learners_0 = empty
Tests_0 = empty

while :
    Learners = non-dominated(Learners)
    Test_gen = generate_tests()
    Learners_gen = generate_learners()
    
    Tests += useful_tests(Test_gen)
    
    Pour chaque L dans Learners_gen:
        Si useful(L, Learners, Tests):
            Learners += L

---

Tests <-> test binaires
            - Appartient aux N premiers
            - On a un vecteur [True False] de la taille du nb d'env

Generation de tests :
            - reproduction de CPPN

Useful Tests :
            - Vecteur [True False] de la taille du nb d'individus caractérisant qui sont ses N premiers
"""

