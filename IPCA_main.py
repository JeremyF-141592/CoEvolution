from Parameters import Configuration
from IPCA.IPCA_core import *
from Utils.Agents import AgentFactory, Agent
from Utils.Environments import EnvironmentInterface
from Utils.Loader import resume_from_folder, prepare_folder
import ipyparallel as ipp
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
if not issubclass(Configuration.baseEnv, EnvironmentInterface):
    raise RuntimeError("Configuration baseEnv is not inherited from EnvironmentInterface.")

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
parser.add_argument('--pop_size', type=int, default=32, help='Population size')
# IPCA
parser.add_argument('--nb_best', type=int, default=8, help='Population size')


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
for t in range(start_from, args.T):
    Learners = generate_learners(Learners, args)

    Learners_gen = generate_learners(Learners, args)

    Tests_gen = generate_tests(Tests)
    Tests = useful(Tests_gen, Tests)

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

