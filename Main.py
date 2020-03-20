from POET.EA_Init import EA_init
from POET.Mutation import Mutate_Envs
from POET.LocalTraining import ES_Step
from POET.Selection import Evaluate_Candidates
from Parameters import agentFactory, baseEnv
from Utils.Agents import AgentFactory, Agent
from Utils.Environment_Interface import EnvironmentInterface
import argparse


# Check Parameters.py --------------------------------------------------------------------------------------------------

if not isinstance(agentFactory, AgentFactory):
    raise RuntimeError("Parameters.py agentFactory is not an instance of AgentFactory.")
if not isinstance(agentFactory.new(), Agent):
    raise RuntimeError("Parameters.py agentFactory.new() is not an instance of Agent.")
if not isinstance(baseEnv, EnvironmentInterface):
    raise RuntimeError("Parameters.py baseEnv is not an instance of EnvironmentInterface.")

# Parse arguments ------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='POET Implementation as in Wang, rui and Lehman, Joel, and Clune, '
                                             'Jeff, and Stanley, Kenneth O. 2019 Uber AI Labs.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

parser.add_argument('--E_init', type=str, default="flat", help='TODO')
parser.add_argument('--Theta_init', type=str, default="random", help='TODO')
parser.add_argument('--Pop_size', type=int, default=10, help='TODO')
parser.add_argument('--Alpha', type=float, default=0.9, help='Learning Rate')
parser.add_argument('--Sigma', type=float, default=1.0, help='Noise std')
parser.add_argument('--T', type=int, default=100, help='TODO')
parser.add_argument('--N_mutate', type=int, default=10, help='TODO')
parser.add_argument('--N_transfer', type=int, default=10, help='TODO')
parser.add_argument('--max_children', type=int, default=10, help='maximum number of children per reproduction')
parser.add_argument('--max_admitted', type=int, default=10, help='maximum number of children admitted per reproduction')
parser.add_argument('--capacity', type=int, default=10, help='maximum number of active environments')
parser.add_argument('--nb_rounds', type=int, default=2, help='Number of rollouts to evaluate one pair')
parser.add_argument('--mc_min', type=int, default=2, help='Minimal number of individual solving an env for the MC')
parser.add_argument('--mc_max', type=int, default=2, help='Maximal number of individual solving an env for the MC')

parser.add_argument('--envs', nargs='+', default=['roughness', 'pit', 'stair', 'stump'])
parser.add_argument('--master_seed', type=int, default=111)

args = parser.parse_args()

# POET Algorithm -------------------------------------------------------------------------------------------------------

EA_List = EA_init(args)
for t in range(args.T):
    if t > 0 and t % args.N_mutate == 0:
        EA_List = Mutate_Envs(EA_List, args)

    M = len(EA_List)
    for m in range(M):
        E, theta = EA_List.items()[m]
        theta = ES_Step(theta, E, args)

    for m in range(M):
        if M > 1 and t % args.N_transfer == 0:
            E, theta = EA_List.items()[m]
            theta_top = Evaluate_Candidates(EA_List[:m] + EA_List[m+1:], E, args)
            if E(theta_top) > E(theta):
                theta = theta_top

