"""
Evolution Strategies (Salimans 2017), evaluated by default on a random set of 20 environments at each iteration.
The environment set can be specified with a pickle file, using --load_env.
"""

from Utils.Loader import resume_from_folder, prepare_folder
from Utils.Stats import bundle_stats, append_stats
from Algorithms.NSGA2.NSGAII_tools import *
from Parameters import Configuration
import ipyparallel as ipp
import argparse
import json
import pickle
import warnings
import os
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

parser = argparse.ArgumentParser(description='Evolution Strategies as in Salimans et al. 2017')

# General
parser.add_argument('--T', type=int, default=400, help='Iterations limit')
parser.add_argument('--resume_from', type=str, default="", help="Resume execution from folder.")
parser.add_argument('--save_to', type=str, default="./ES_execution", help="Execution save-to folder.")
parser.add_argument('--save_mode', type=str, default="all", help="Specify save mode among ['all', 'last', N] where N is"
                                                                 "a number corresponding the saving's interval.")
parser.add_argument('--verbose', type=int, default=0, help="Print information.")
parser.add_argument('--max_budget', type=int, default=-1, help="Maximum number of environment evaluations.")
# Population
parser.add_argument('--pop_size', type=int, default=100, help='Population size')
parser.add_argument('--pop_env_size', type=int, default=20, help='Environment Population size')
parser.add_argument('--load_env', type=str, default="", help='Path to pickled environment')
# Local optimization
parser.add_argument('--lr_init', type=float, default=0.01, help="Learning rate initial value")
parser.add_argument('--lr_decay', type=float, default=0.9999, help="Learning rate decay")
parser.add_argument('--lr_limit', type=float, default=0.001, help="Learning rate limit")
parser.add_argument('--noise_std', type=float, default=0.1,  help='Noise std for local ES-optimization')
parser.add_argument('--noise_decay', type=float, default=0.999)
parser.add_argument('--noise_limit', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for ES gradient descent')
parser.add_argument('--w_decay', type=float, default=0.01, help='Weight decay penalty')

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


def ES_Step(theta, envs, args):
    """Local optimization by Evolution Strategy steps, rank normalization and weight decay."""
    og_weights = theta.get_weights()

    shared_gaussian_table = [np.random.normal(0, 1, size=len(og_weights)) for i in range(args.batch_size)]

    if theta.get_opt_state() is None:
        theta.set_opt_state(Configuration.optimizer.default_state())
    if "t" not in theta.get_opt_state().keys():
        z = theta.get_opt_state().copy()
        z.update({"t": 1})
        theta.set_opt_state(z)

    sigma = max(args.noise_limit, args.noise_std * args.noise_decay ** theta.get_opt_state()["t"])

    thetas = []
    for i in range(args.batch_size):
        new_theta = Configuration.agentFactory.new()
        new_theta.set_weights(og_weights + sigma * shared_gaussian_table[i])
        thetas.append(new_theta)

    scores = list()
    for E in envs:
        partial_scores = Configuration.lview.map(E, thetas)
        if len(scores) == 0:
            scores = partial_scores.copy()
        else:
            for i in range(len(scores)):
                scores[i] += partial_scores[i]
        Configuration.budget_spent[-1] += len(thetas)
    scores = np.array(scores)

    self_score = 0
    for E in envs:
        self_score += E(theta)
    self_score /= len(envs)

    for i in range(len(scores)):
        scores[i] -= args.w_decay * np.linalg.norm(og_weights + sigma * shared_gaussian_table[i])

    scores = rank_normalize(scores)

    summed_weights = np.zeros(og_weights.shape)
    for i in range(len(scores)):
        summed_weights += scores[i] * shared_gaussian_table[i]
    grad_estimate = -(1/(len(shared_gaussian_table))) * summed_weights

    step, new_state = Configuration.optimizer.step(grad_estimate, theta.get_opt_state())

    new_ag = Configuration.agentFactory.new()
    new_ag.set_opt_state(new_state)
    new_ag.set_weights(og_weights + step)
    return new_ag, self_score


def rank_normalize(arr):
    asorted = arr.argsort()
    linsp = np.linspace(0, 1, num=len(asorted))
    res = np.zeros(len(asorted))
    for i in range(len(asorted)):
        res[asorted[i]] = linsp[i]
    return 2*res - 1


envs = list()

default = True
if os.path.exists(args.load_env):
    with open(args.load_env, "rb") as f:
        envs = pickle.load(f)
    default = False

# ES Algorithm ---------------------------------------------------------------------------------------------------------

if len(pop) == 0:
    pop.append(Configuration.agentFactory.new())

for t in range(start_from, args.T):
    print(f"Iteration {t} ...", flush=True)
    Configuration.budget_spent.append(0)

    if default:
        envs = list()
        for i in range(args.pop_env_size):
            ev = Configuration.envFactory.new()
            for j in range(30):
                ev = ev.get_child()
            envs.append(ev)

    ag, sc = ES_Step(pop[0], envs, args)

    pop = [ag]

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
    bundle["Fitness"] = sc
    append_stats(f"{args.save_to}/Stats.json", bundle)
    if args.verbose > 0:
        print(f"\tExecution saved at {args.save_to}.")
    if 0 < args.max_budget < sum(Configuration.budget_spent):
        print(f"\nMaximum budget exceeded : {sum(Configuration.budget_spent)} > {args.max_budget}.\n")
        break
