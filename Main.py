from Parameters import Configuration
from POET.EA_Init import ea_init
from POET.Mutation import mutate_envs
from POET.LocalTraining import ES_Step
from POET.Selection import Evaluate_Candidates
from Utils.Agents import AgentFactory, Agent
from Utils.Environments import EnvironmentInterface
import ipyparallel as ipp
import argparse
import json
import pickle
import os
from glob import glob


Configuration.make()
# Ipyparallel --------------------------------------------------------------------------------------------------
# Local parallelism
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
if not issubclass(Configuration.baseEnv, EnvironmentInterface):
    raise RuntimeError("Configuration baseEnv is not inherited from EnvironmentInterface.")

# Parse arguments ------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='POET Implementation as in Wang, rui and Lehman, Joel, and Clune, '
                                             'Jeff, and Stanley, Kenneth O. 2019 Uber AI Labs.')

parser.add_argument('--E_init', type=str, default="flat", help='Initial policy of environments among ["flat"]')
parser.add_argument('--Theta_init', type=str, default="random", help='Initial policy of individuals among ["random"]')
parser.add_argument('--Pop_size', type=int, default=2, help='Population size')
parser.add_argument('--alpha', type=float, default=0.01, help='Learning Rate for local ES-optimization')
parser.add_argument('--sigma', type=float, default=0.1, help='Noise std for local ES-optimization')
parser.add_argument('--T', type=int, default=100, help='Iterations limit')
parser.add_argument('--N_mutate', type=int, default=2, help='Number of steps before attempting mutation')
parser.add_argument('--N_transfer', type=int, default=2, help='Number of steps before attempting transfer')
parser.add_argument('--max_children', type=int, default=10, help='maximum number of children per reproduction')
parser.add_argument('--max_admitted', type=int, default=10, help='maximum number of children admitted per reproduction')
parser.add_argument('--capacity', type=int, default=10, help='maximum number of active environments - REPLACED'
                                                             'by Pop_size.')
parser.add_argument('--nb_rounds', type=int, default=1, help='Number of rollouts to evaluate one pair')
parser.add_argument('--mc_min', type=int, default=-198, help='Minimal number of individual solving an env for the MC')
parser.add_argument('--mc_max', type=int, default=300, help='Maximal number of individual solving an env for the MC')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for ES gradient descent')

parser.add_argument('--resume', type=str, default="", help="Resume execution from folder.")
parser.add_argument('--save_to', type=str, default="./POET_execution", help="Execution save-to folder.")

parser.add_argument('--envs', nargs='+', default=['roughness', 'pit', 'stair', 'stump'])
parser.add_argument('--master_seed', type=int, default=111)

args = parser.parse_args()

# Resume execution -----------------------------------------------------------------------------------------------------

resume = ""
resume_from = 0
ea_list_resume = []
if args.resume != "":
    resume = args.resume

if resume != "":
    with open(resume + "/commandline_args.txt", 'r') as f:
        args.__dict__ = json.load(f)
    filenames = glob(f"{resume}/*.pickle")
    filenames.sort()
    # assume we only have relevant files in the folder, take the last sorted .pickle file
    ea_path = filenames[-1]
    resume_from = len(filenames)
    with open(f"{ea_path}", "rb") as f:
        ea_list_resume = pickle.load(f)
    print(f"Execution successfully resumed from {resume} .")
else:
    if not os.path.exists(args.save_to):
        os.mkdir(args.save_to)
    with open(f"{args.save_to}/commandline_args.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)


# POET Algorithm -------------------------------------------------------------------------------------------------------

EA_List = ea_init(args) if resume == "" else ea_list_resume
for t in range(resume_from, args.T):
    print(f"Iteration {t} ...", end=" ")

    if t > 0 and t % args.N_mutate == 0:
        print("Mutate ...", end=" ")
        EA_List = mutate_envs(EA_List, args)

    M = len(EA_List)
    for m in range(M):
        E, theta = EA_List[m]
        ES_Step(theta, E, args, in_place=True)  # Operates in-place to lighten execution

    top_debug = -200
    if M > 1 and t > 0 and t % args.N_transfer == 0:
        print("Transfer ...", end=" ")
        for m in range(M):
            E, theta = EA_List[m]
            theta_top = Evaluate_Candidates(EA_List[:m] + EA_List[m+1:], E, args)
            debug_top = E(theta_top)
            top_debug = max(top_debug, debug_top)
            if E(theta_top) > E(theta):
                theta = theta_top

        print("\t Best current score (debug) :", top_debug)
    print("Done.")
    with open(f'{args.save_to}/Iteration {t}.pickle', 'wb') as f:
        pickle.dump(EA_List, f)
